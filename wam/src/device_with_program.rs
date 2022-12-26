use super::{
    log_debug, log_info, log_trace,
    machine::{Instructions, SystemCallEncoder, SystemCalls},
    CodeSubmission, Command, GetSystemCallsResponse, IoError, ReportStatusResponse,
    SerialConnection, SerialReadWrite,
};

type EscalatedCommand = super::device_with_query::UnhandledCommand;

#[derive(Debug)]
enum HandledCommand {
    ReportStatus,
    GetSystemCalls,
    SubmitQuery { code_submission: CodeSubmission },
}

#[derive(Debug)]
pub enum UnhandledCommand {
    Reset,
    SubmitProgram { code_submission: CodeSubmission },
}

#[derive(Debug)]
enum Action {
    ProcessNextCommand,
    ProcessCommand(HandledCommand),
    Escalate(UnhandledCommand),
}

pub struct Device<'m, 's, Serial: SerialReadWrite, Calls> {
    pub program: Instructions<'m>,
    pub memory: &'m mut [u32],
    pub serial_connection: &'s mut SerialConnection<Serial>,
    pub system_calls: &'s mut Calls,
}

impl<'m, 's, Serial: SerialReadWrite, Calls: SystemCalls> Device<'m, 's, Serial, Calls> {
    fn handle_submit_query(
        &mut self,
        code_submission: CodeSubmission,
    ) -> Result<Action, IoError<Serial>> {
        log_info!("Loading query");

        let super::LoadedCode {
            code_section: query,
            rest_of_memory: memory,
        } = match super::load_code(self.memory, self.serial_connection, code_submission)? {
            Ok(code) => code,
            Err(error) => {
                self.serial_connection.encode(super::ErrorResponse(error))?;
                return Ok(Action::ProcessNextCommand);
            }
        };

        let unhandled_command = super::device_with_query::Device {
            program: self.program,
            query,
            memory,
            serial_connection: self.serial_connection,
            system_calls: self.system_calls,
        }
        .run()?;

        log_trace!("Finished with query");

        Ok(match unhandled_command {
            EscalatedCommand::Reset => Action::Escalate(UnhandledCommand::Reset),
            EscalatedCommand::ProcessNextCommand => Action::ProcessNextCommand,
            EscalatedCommand::SubmitProgram { code_submission } => {
                Action::Escalate(UnhandledCommand::SubmitProgram { code_submission })
            }
            EscalatedCommand::SubmitQuery { code_submission } => {
                Action::ProcessCommand(HandledCommand::SubmitQuery { code_submission })
            }
        })
    }

    fn process_command(&mut self, command: HandledCommand) -> Result<Action, IoError<Serial>> {
        match command {
            HandledCommand::ReportStatus => {
                let status = ReportStatusResponse::WaitingForQuery;
                log_debug!("Status: {status}");
                self.serial_connection.encode(status)?;
                Ok(Action::ProcessNextCommand)
            }
            HandledCommand::GetSystemCalls => {
                self.serial_connection
                    .encode(GetSystemCallsResponse::SystemCalls(SystemCallEncoder(
                        self.system_calls,
                    )))?;
                Ok(Action::ProcessNextCommand)
            }
            HandledCommand::SubmitQuery { code_submission } => {
                self.handle_submit_query(code_submission)
            }
        }
    }

    pub(crate) fn run(mut self) -> Result<UnhandledCommand, IoError<Serial>> {
        log_info!("Waiting for Query");
        loop {
            let command = self.serial_connection.decode::<super::Command>()?;

            log_info!("Processing command {:?}", command);

            let mut command = match command {
                Command::ReportStatus => HandledCommand::ReportStatus,
                Command::GetSystemCalls => HandledCommand::GetSystemCalls,
                Command::SubmitProgram { code_submission } => {
                    return Ok(UnhandledCommand::SubmitProgram { code_submission })
                }
                Command::SubmitQuery { code_submission } => {
                    HandledCommand::SubmitQuery { code_submission }
                }
                Command::LookupMemory { .. } | Command::NextSolution => {
                    return Ok(UnhandledCommand::Reset)
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
