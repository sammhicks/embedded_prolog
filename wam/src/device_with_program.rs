use super::{
    log_debug, log_info, log_trace, CommandHeader, ProcessInputError, SerialConnection, SerialRead,
    SerialWrite,
};

type EscalatedCommand = super::device_with_query::UnhandledCommand;

#[derive(Debug)]
enum HandledCommand {
    ReportStatus,
    SubmitQuery,
}

#[derive(Debug)]
pub enum UnhandledCommand {
    Reset,
    SubmitProgram,
}

#[derive(Debug)]
enum Action {
    ProcessNextCommand,
    ProcessCommand(HandledCommand),
    Escalate(UnhandledCommand),
}

pub struct Device<'a, 's, S> {
    pub program: &'a [u32],
    pub memory: &'a mut [u32],
    pub serial_connection: &'s mut SerialConnection<S>,
}

impl<'a, 's, S: SerialRead<u8> + SerialWrite<u8>> Device<'a, 's, S> {
    fn handle_submit_query(&mut self) -> Result<Action, ProcessInputError> {
        log_info!("Loading query");

        let super::LoadedCode {
            code_section: query,
            rest_of_memory: memory,
        } = match super::load_code(&mut self.memory, &mut self.serial_connection)? {
            Some(c) => c,
            None => return Ok(Action::ProcessNextCommand),
        };

        let unhandled_command = super::device_with_query::Device {
            program: self.program,
            query,
            memory,
            serial_connection: &mut self.serial_connection,
        }
        .run()?;

        log_trace!("Finished with query");

        Ok(match unhandled_command {
            EscalatedCommand::Reset => Action::Escalate(UnhandledCommand::Reset),
            EscalatedCommand::ProcessNextCommand => Action::ProcessNextCommand,
            EscalatedCommand::SubmitProgram => Action::Escalate(UnhandledCommand::SubmitProgram),
            EscalatedCommand::SubmitQuery => Action::ProcessCommand(HandledCommand::SubmitQuery),
        })
    }

    fn process_command(&mut self, command: HandledCommand) -> Result<Action, ProcessInputError> {
        match command {
            HandledCommand::ReportStatus => {
                log_debug!("Status: Waiting for Query");
                self.serial_connection.write_single_char('Q')?;
                Ok(Action::ProcessNextCommand)
            }
            HandledCommand::SubmitQuery => self.handle_submit_query(),
        }
    }

    pub fn run(mut self) -> Result<UnhandledCommand, ProcessInputError> {
        log_info!("Waiting for Query");
        loop {
            let command = CommandHeader::parse(self.serial_connection.read_ascii_char()?)?;

            log_info!("Processing command {:?}", command);

            let mut command = match command {
                CommandHeader::ReportStatus => HandledCommand::ReportStatus,
                CommandHeader::SubmitProgram => return Ok(UnhandledCommand::SubmitProgram),
                CommandHeader::SubmitQuery => HandledCommand::SubmitQuery,
                CommandHeader::LookupMemory | CommandHeader::NextSolution => {
                    return Err(ProcessInputError::Unexpected(command as u8))
                }
            };

            loop {
                match self.process_command(command)? {
                    Action::ProcessNextCommand => break,
                    Action::ProcessCommand(next_command) => {
                        command = next_command;
                    }
                    Action::Escalate(unhandled_command) => return Ok(unhandled_command),
                }
            }
        }
    }
}
