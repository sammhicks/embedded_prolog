use super::{
    load_code, log_debug, log_info, log_trace, machine::SystemCalls, CommandHeader, IoError,
    LoadedCode, Never, ProcessInputError, SerialConnection, SerialRead, SerialWrite, SUCCESS,
};

enum Action {
    ProcessNextCommand,
    ProcessCommand(CommandHeader),
}

pub struct Device<'a, Serial, Calls> {
    pub memory: &'a mut [u32],
    pub serial_connection: SerialConnection<Serial>,
    pub system_calls: Calls,
}

impl<'a, Serial: SerialRead<u8> + SerialWrite<u8>, Calls: SystemCalls> Device<'a, Serial, Calls> {
    fn handle_submit_program(&mut self) -> Result<Action, ProcessInputError> {
        log_info!("Loading program");

        self.serial_connection
            .write_system_calls(&self.system_calls)?;

        let LoadedCode {
            code_section: program,
            rest_of_memory: memory,
        } = match load_code(self.memory, &mut self.serial_connection)? {
            Some(c) => c,
            None => return Ok(Action::ProcessNextCommand),
        };

        self.serial_connection.write_single_char(SUCCESS)?;

        let unhandled_command = crate::device_with_program::Device {
            program,
            memory,
            serial_connection: &mut self.serial_connection,
            system_calls: &mut self.system_calls,
        }
        .run()?;

        log_trace!("Finished with program");

        match unhandled_command {
            crate::device_with_program::UnhandledCommand::SubmitProgram => {
                Ok(Action::ProcessCommand(CommandHeader::SubmitProgram))
            }
            crate::device_with_program::UnhandledCommand::Reset => Ok(Action::ProcessNextCommand),
        }
    }

    fn process_command_byte(&mut self, command_byte: u8) -> Result<Action, ProcessInputError> {
        let Ok(command_header) = CommandHeader::parse(command_byte) else {
            crate::error!(
                &mut self.serial_connection,
                "Invalid command: {:?}",
                command_byte as char
            )?;
            return Ok(Action::ProcessNextCommand);
        };

        log_info!("Processing command {:?}", command_header);

        match command_header {
            CommandHeader::ReportStatus => {
                log_debug!("Status: Waiting for program");
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
        log_info!("Running");
        let mut command_header = self.serial_connection.read_ascii_char()?;
        loop {
            let next_command_header = match self.process_command_byte(command_header) {
                Ok(Action::ProcessNextCommand) => self.serial_connection.read_ascii_char()?,
                Ok(Action::ProcessCommand(command)) => command as u8,
                Err(ProcessInputError::Unexpected(b)) => b,
                Err(ProcessInputError::IoError) => return Err(IoError),
            };

            command_header = next_command_header;
        }
    }
}
