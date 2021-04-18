use super::{
    load_code, CommandHeader, IoError, LoadedCode, Never, ProcessInputError, SerialConnection,
    SerialRead, SerialWrite, SUCCESS,
};

enum Action {
    ProcessNextCommand,
    ProcessCommand(CommandHeader),
}

pub struct Device<'a, S> {
    pub memory: &'a mut [u32],
    pub serial_connection: SerialConnection<S>,
}

impl<'a, S: SerialRead<u8> + SerialWrite<u8>> Device<'a, S> {
    fn handle_submit_program(&mut self) -> Result<Action, ProcessInputError> {
        crate::log_info!("Loading program");

        let LoadedCode {
            code_section: program,
            rest_of_memory: memory,
        } = match load_code(&mut self.memory, &mut self.serial_connection)? {
            Some(c) => c,
            None => return Ok(Action::ProcessNextCommand),
        };

        self.serial_connection.write_single_char(SUCCESS)?;

        let unhandled_command = crate::device_with_program::Device {
            program,
            memory,
            serial_connection: &mut self.serial_connection,
        }
        .run()?;

        crate::log_trace!("Finished with program");

        match unhandled_command {
            crate::device_with_program::UnhandledCommand::SubmitProgram => {
                Ok(Action::ProcessCommand(CommandHeader::SubmitProgram))
            }
            crate::device_with_program::UnhandledCommand::Reset => Ok(Action::ProcessNextCommand),
        }
    }

    fn process_command_byte(&mut self, command_byte: u8) -> Result<Action, ProcessInputError> {
        let command_header = match CommandHeader::parse(command_byte) {
            Ok(header) => header,
            Err(_) => {
                crate::error!(
                    &mut self.serial_connection,
                    "Invalid command: {:?}",
                    command_byte as char
                )?;
                return Ok(Action::ProcessNextCommand);
            }
        };

        crate::log_debug!("Processing command {:?}", command_header);

        match command_header {
            CommandHeader::ReportStatus => {
                log::debug!("Status: Waiting for program");
                self.serial_connection.write_single_char('P')?;
                Ok(Action::ProcessNextCommand)
            }
            CommandHeader::SubmitProgram => self.handle_submit_program(),
            CommandHeader::SubmitQuery
            | CommandHeader::LookupMemory
            | CommandHeader::NextSolution => {
                crate::error!(
                    &mut self.serial_connection,
                    "Cannot handle {:?}: No program",
                    command_header
                )?;
                Ok(Action::ProcessNextCommand)
            }
        }
    }

    pub fn run(mut self) -> Result<Never, IoError> {
        crate::log_info!("Running");
        let mut command_header = self.serial_connection.read_ascii_char()?;
        loop {
            let next_command_header = match self.process_command_byte(command_header) {
                Ok(Action::ProcessNextCommand) => self.serial_connection.read_ascii_char()?,
                Ok(Action::ProcessCommand(command)) => command as u8,
                Err(ProcessInputError::Unexpected(b)) => b,
                Err(ProcessInputError::ReadError) => return Err(IoError),
            };

            command_header = next_command_header;
        }
    }
}
