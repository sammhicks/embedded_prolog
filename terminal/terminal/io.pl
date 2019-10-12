:- module(io, [
              read_nonspace/2,
              read_u8/2,
              read_u32/2,
              write_words/2
          ]).

:- use_module(library(apply)).


read_nonspace(Stream, Result) :-
    get_char(Stream, C),
    (   char_type(C, space)
    ->  read_nonspace(Stream, Result)
    ;   Result = C
    ).


read_u8(Stream, N) :-
    read_un(Stream, 1, N).


read_u32(Stream, N) :-
    read_un(Stream, 4, N).


read_un(Stream, Count, N) :-
    Char_Count is 2 * Count,
    length(Hex_Chars, Char_Count),
    maplist(get_char(Stream), Hex_Chars),
    hex_bytes(Hex_Chars, Bytes),
    foldl(be_bytes, Bytes, 0, N).


be_bytes(Byte, Acc, Next) :-
    Next is 256 * Acc + Byte.


write_words(Stream, Words) :-
    length(Words, Length),
    write_u32(Stream, Length),
    maplist(write_word(Stream), Words).


write_word(Stream, Word) :-
    maplist(write_u8(Stream), Word).


write_u8(Stream, N) :-
    format(Stream, "~|~`0t~16r~2+", [N]).


write_u32(Stream, N) :-
    format(Stream, "~|~`0t~16r~8+", [N]).
