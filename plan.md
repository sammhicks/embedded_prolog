Commands
+ S - Report Status
  + P - Waiting for Program
  + Q - Waiting for Query
  + R - Executing Program
  + A - Single Answer
  + C - Multiple Answers
  + F - No Answers
+ P - Submit Program
  + length in words(u32): BE u32 in hex
  + program words in BE hex
  + SHA256 hash in BE hex
  + Response on success: 'S'
+ Q - Run Query
  + length in words(u32): BE u32 in hex
  + query words in BE hex
  + SHA256 hash in BE hex
  + Response on success: 'S'

States
+ WaitingForProgram
+ WaitingForQuery
+ ExecutingProgram
+ SingleAnswer
+ ChoicePoint

Messages:
    Header, u32 length in hex, then bytes in hex
    Headers:
        + I - Info
        + E - Error
    After Messages: 'S'
