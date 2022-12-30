use super::{
    log_debug, log_info,
    machine::{Address, ExecutionFailure, Instructions, SystemCallEncoder, SystemCalls, TermsList},
    ErrorResponse, IoError, SerialConnection, SerialReadWrite,
};

#[derive(Debug)]
pub enum UnhandledCommand {
    Reset,
    ProcessNextCommand,
    SubmitProgram {
        code_submission: comms::CodeSubmission,
    },
    SubmitQuery {
        code_submission: comms::CodeSubmission,
    },
}

type SubmitQueryResponse<'a> = comms::SubmitQueryResponse<TermsList<'a>>;

pub struct Device<'m, 's, Serial: SerialReadWrite, Calls> {
    pub program: Instructions<'m>,
    pub query: Instructions<'m>,
    pub memory: &'m mut [u32],
    pub serial_connection: &'s mut SerialConnection<Serial>,
    pub system_calls: &'s mut Calls,
}

impl<'m, 's, Serial: SerialReadWrite, Calls: SystemCalls> Device<'m, 's, Serial, Calls> {
    pub(crate) fn run(self) -> Result<UnhandledCommand, IoError<Serial>> {
        let mut machine =
            crate::machine::Machine::new(self.program, self.query, self.memory, self.system_calls);

        let mut execution_result = machine.next_solution();

        loop {
            match execution_result {
                Ok(()) => (),
                Err(ExecutionFailure::Failed) => {
                    self.serial_connection
                        .encode(SubmitQueryResponse::NoSolution)?;
                    return Ok(UnhandledCommand::ProcessNextCommand);
                }
                Err(ExecutionFailure::Error(error)) => {
                    self.serial_connection.encode(ErrorResponse(error))?;
                    return Ok(UnhandledCommand::Reset);
                }
            }

            match machine.solution() {
                Ok(solution) => self
                    .serial_connection
                    .encode(SubmitQueryResponse::Solution(solution.into_comms()))?,
                Err(err) => {
                    self.serial_connection.encode(ErrorResponse(err))?;
                    return Ok(UnhandledCommand::Reset);
                }
            }

            execution_result = loop {
                let command = self.serial_connection.decode::<comms::Command>()?;

                log_info!("Processing command {:?}", command);

                match command {
                    comms::Command::ReportStatus => {
                        let status = comms::ReportStatusResponse::WaitingForQuery;
                        log_debug!("Status: {:?}", status);
                        self.serial_connection.encode(status)?;
                    }
                    comms::Command::GetSystemCalls => {
                        self.serial_connection.encode(
                            comms::GetSystemCallsResponse::SystemCalls(SystemCallEncoder(
                                machine.system_calls(),
                            )),
                        )?;
                    }
                    comms::Command::SubmitProgram { code_submission } => {
                        return Ok(UnhandledCommand::SubmitProgram { code_submission })
                    }
                    comms::Command::SubmitQuery { code_submission } => {
                        return Ok(UnhandledCommand::SubmitQuery { code_submission })
                    }
                    comms::Command::LookupMemory { address } => {
                        match machine.lookup_memory(Address::new(address)) {
                            Ok((address, value)) => {
                                self.serial_connection.encode(
                                    comms::LookupMemoryResponse::MemoryValue {
                                        address: address.into_inner(),
                                        value: value.into_comms(),
                                    },
                                )?;
                            }
                            Err(error) => {
                                self.serial_connection.encode(ErrorResponse(error))?;
                                return Ok(UnhandledCommand::Reset);
                            }
                        };
                    }
                    comms::Command::NextSolution => {
                        break machine.next_solution();
                    }
                }
            };
        }
    }
}
