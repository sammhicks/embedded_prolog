use core::fmt;

use super::{
    load_code, log_debug, log_info, log_trace,
    machine::{SystemCallEncoder, SystemCalls},
    ErrorResponse, IoError, LoadedCode, Never, SerialConnection, SerialReadWrite,
};

#[derive(Debug)]
enum HandledCommand {
    ReportStatus,
    GetSystemCalls,
    SubmitProgram {
        code_submission: comms::CodeSubmission,
    },
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
enum UnhandledCommand {
    SubmitQuery,
    LookupMemory,
    NextSolution,
}

#[cfg_attr(feature = "defmt-logging", derive(defmt::Format))]
struct NoProgramError {
    unhandled_command: UnhandledCommand,
}

impl fmt::Display for NoProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cannot handle {:?}: No Program", self.unhandled_command)
    }
}

enum Action {
    Reset,
    ProcessNextCommand,
    ProcessCommand(comms::Command),
}

pub enum Error<S: SerialReadWrite> {
    Reset,
    IoError(IoError<S>),
}

impl<S: SerialReadWrite> From<IoError<S>> for Error<S> {
    fn from(inner: IoError<S>) -> Self {
        Self::IoError(inner)
    }
}

impl<S: SerialReadWrite> fmt::Debug for Error<S>
where
    S::Error: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Reset => "Reset".fmt(f),
            Error::IoError(inner) => f.debug_tuple("IoError").field(inner).finish(),
        }
    }
}

impl<S: SerialReadWrite> fmt::Display for Error<S>
where
    S::Error: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Reset => "Reset".fmt(f),
            Error::IoError(inner) => inner.fmt(f),
        }
    }
}

pub struct Device<'a, Serial: SerialReadWrite, Calls> {
    pub memory: &'a mut [u32],
    pub serial_connection: SerialConnection<Serial>,
    pub system_calls: Calls,
}

impl<'a, Serial: SerialReadWrite, Calls: SystemCalls> Device<'a, Serial, Calls> {
    fn handle_submit_program(
        &mut self,
        code_submission: comms::CodeSubmission,
    ) -> Result<Action, IoError<Serial>> {
        log_info!("Loading program");

        let LoadedCode {
            code_section: program,
            rest_of_memory: memory,
        } = match load_code(self.memory, &mut self.serial_connection, code_submission)? {
            Ok(code) => {
                self.serial_connection
                    .encode(comms::SubmitProgramResponse::Success)?;

                code
            }
            Err(err) => {
                self.serial_connection.encode(ErrorResponse(err))?;
                return Ok(Action::ProcessNextCommand);
            }
        };

        let unhandled_command = crate::device_with_program::Device {
            program,
            memory,
            serial_connection: &mut self.serial_connection,
            system_calls: &mut self.system_calls,
        }
        .run()?;

        log_trace!("Finished with program");

        match unhandled_command {
            crate::device_with_program::UnhandledCommand::SubmitProgram { code_submission } => {
                Ok(Action::ProcessCommand(comms::Command::SubmitProgram {
                    code_submission,
                }))
            }
            crate::device_with_program::UnhandledCommand::Reset => Ok(Action::Reset),
        }
    }

    fn process_command(&mut self, command: comms::Command) -> Result<Action, IoError<Serial>> {
        log_info!("Processing command {:?}", command);

        let command = match command {
            comms::Command::ReportStatus => Ok(HandledCommand::ReportStatus),
            comms::Command::GetSystemCalls => Ok(HandledCommand::GetSystemCalls),
            comms::Command::SubmitProgram { code_submission } => {
                Ok(HandledCommand::SubmitProgram { code_submission })
            }
            comms::Command::SubmitQuery { .. } => Err(UnhandledCommand::SubmitQuery),
            comms::Command::LookupMemory { .. } => Err(UnhandledCommand::LookupMemory),
            comms::Command::NextSolution => Err(UnhandledCommand::NextSolution),
        };

        match command {
            Ok(HandledCommand::ReportStatus) => {
                let status = comms::ReportStatusResponse::WaitingForProgram;
                log_debug!("Status: {:?}", status);
                self.serial_connection.encode(status)?;
                Ok(Action::ProcessNextCommand)
            }
            Ok(HandledCommand::GetSystemCalls) => {
                self.serial_connection
                    .encode(comms::GetSystemCallsResponse::SystemCalls(
                        SystemCallEncoder(&self.system_calls),
                    ))?;
                Ok(Action::ProcessNextCommand)
            }
            Ok(HandledCommand::SubmitProgram { code_submission }) => {
                self.handle_submit_program(code_submission)
            }
            Err(unhandled_command) => {
                self.serial_connection
                    .encode(ErrorResponse(NoProgramError { unhandled_command }))?;
                Ok(Action::Reset)
            }
        }
    }

    pub fn run(mut self) -> Result<Never, Error<Serial>> {
        log_info!("Running");
        let mut command = self.serial_connection.decode::<comms::Command>()?;
        loop {
            command = match self.process_command(command)? {
                Action::Reset => return Err(Error::Reset),
                Action::ProcessNextCommand => self.serial_connection.decode::<comms::Command>()?,
                Action::ProcessCommand(command) => command,
            };
        }
    }
}
