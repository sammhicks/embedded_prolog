use super::{
    log_debug, log_info, log_trace, machine::Instructions, machine::SystemCalls, CommandHeader,
    ProcessInputError, SerialConnection, SerialRead, SerialWrite,
};

use crate::machine::{ExecutionFailure, OptionDisplay};

#[derive(Debug)]
pub enum UnhandledCommand {
    Reset,
    ProcessNextCommand,
    SubmitProgram,
    SubmitQuery,
}

pub struct Device<'m, 's, Serial, Calls> {
    pub program: Instructions<'m>,
    pub query: Instructions<'m>,
    pub memory: &'m mut [u32],
    pub serial_connection: &'s mut SerialConnection<Serial>,
    pub system_calls: &'s mut Calls,
}

impl<'m, 's, Serial: SerialRead<u8> + SerialWrite<u8>, Calls: SystemCalls>
    Device<'m, 's, Serial, Calls>
{
    pub fn run(self) -> Result<UnhandledCommand, ProcessInputError> {
        let mut machine =
            crate::machine::Machine::new(self.program, self.query, self.memory, self.system_calls);

        let mut execution_result = machine.next_solution();

        loop {
            let success = match execution_result {
                Ok(success) => success,
                Err(ExecutionFailure::Failed) => {
                    self.serial_connection.write_single_char('F')?;
                    return Ok(UnhandledCommand::ProcessNextCommand);
                }
                Err(ExecutionFailure::Error(err)) => {
                    crate::error!(self.serial_connection, "{:?}", err)?;
                    return Ok(UnhandledCommand::Reset);
                }
            };

            let solution_registers = match machine.solution_registers() {
                Ok(registers) => registers,
                Err(err) => {
                    crate::error!(self.serial_connection, "{:?}", err)?;
                    return Ok(UnhandledCommand::Reset);
                }
            };

            let success_code = match success {
                crate::machine::ExecutionSuccess::SingleAnswer => 'A',
                crate::machine::ExecutionSuccess::MultipleAnswers => 'C',
            };

            self.serial_connection.write_char(success_code)?;
            for address in solution_registers {
                log_trace!("Solution Register: {}", OptionDisplay(address));
                self.serial_connection.write_be_serializable_hex(address)?;
            }
            self.serial_connection.flush()?;

            execution_result = loop {
                let command = CommandHeader::parse(self.serial_connection.read_ascii_char()?)?;

                log_info!("Processing command {:?}", command);

                match command {
                    CommandHeader::ReportStatus => {
                        log_debug!("Status: {:?}", success);
                        self.serial_connection.write_single_char(success_code)?;
                    }
                    CommandHeader::SubmitProgram => return Ok(UnhandledCommand::SubmitProgram),
                    CommandHeader::SubmitQuery => return Ok(UnhandledCommand::SubmitQuery),
                    CommandHeader::LookupMemory => {
                        match machine
                            .lookup_memory(self.serial_connection.read_be_serializable_hex()?)
                        {
                            Ok((address, value, data, subterms)) => {
                                self.serial_connection
                                    .write_value(address, value, data, subterms)?;
                                self.serial_connection.flush()?;
                            }
                            Err(err) => {
                                crate::error!(self.serial_connection, "{:?}", err)?;
                                return Ok(UnhandledCommand::Reset);
                            }
                        };
                    }
                    CommandHeader::NextSolution => {
                        break machine.next_solution();
                    }
                }
            };
        }
    }
}
