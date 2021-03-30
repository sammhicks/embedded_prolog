:- module(http_terminal, [
              connect/0
       ]).

:- use_module(library(socket)).

:- use_module(terminal/terminal).

:- export(disconnect/0).
:- export(status/1).
:- export(compile_program/2).
:- export(run_query/2).


connect :-
    disconnect,
    tcp_connect(localhost:8080, Stream, []),
    asserta(current_connection(Stream)).

