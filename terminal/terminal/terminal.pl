:- module(terminal, [
              current_connection/1,
              disconnect/0,
              status/1,
              compile_program/0
          ]).

:- use_module(io).

:- dynamic(current_connection/1).

disconnect :-
    (   current_connection(Stream)
    ->  close(Stream)
    ;   true),
    retractall(current_connection(_)).


status(Status) :-
    (   current_connection(Stream)
    ->  write(Stream, 'S'),
        flush_output(Stream),
        read_nonspace(Stream, Header),
        char_status(Stream, Header, Status),
        !
    ;   Status = disconnected).



char_status(_, 'P', waiting_for_program).
char_status(_, 'Q', waiting_for_query).
char_status(Stream, 'E', error(Error)) :-
    read_error(Stream, Error).


read_error(Stream, Chars) :-
    read_error_chars(Stream, Chars).


read_error_chars(Stream, Chars) :-
    read_u32(Stream, Count),
    length(Section, Count),
    maplist(read_u8(Stream), Section),
    append(Section, Rest, Chars),
    read_nonspace(Stream, C),
    (   C == 'E'
    ->  read_error_chars(Stream, Rest)
    ;   C == 'S'
    ->  Rest = []
    ).


compile_program.
