:- module(io, [
              read_nonspace/2,
              try_read_u8/2,
              read_u8/2,
              read_u16/2,
              read_u32/2,
              decode_be_bytes/2,
              write_u16/2,
              write_words_with_hash/2
          ]).

:- use_module(library(apply)).
:- use_module(library(sha)).


read_nonspace(Stream, Result) :-
    get_char(Stream, C),
    (   char_type(C, space)
    ->  read_nonspace(Stream, Result)
    ;   Result = C
    ).


try_read_u8(Stream, Result) :-
    get_code(Stream, AC),
    (   is_digit(AC, 16, A)
    ->  get_code(Stream, BC),
        is_digit(BC, 16, B),
        N is 16 * A + B,
        Result = ok(N)
    ;   char_code(E, AC),
        Result = err(E)
    ),
    !.


read_u8(Stream, N) :-
    read_un(Stream, 1, N).


read_u16(Stream, N) :-
    read_un(Stream, 2, N).


read_u32(Stream, N) :-
    read_un(Stream, 4, N).


read_un(Stream, Count, N) :-
    Char_Count is 2 * Count,
    length(Hex_Chars, Char_Count),
    maplist(get_char(Stream), Hex_Chars),
    hex_bytes(Hex_Chars, Bytes),
    decode_be_bytes(Bytes, N).


decode_be_bytes(Bytes, N) :-
    foldl(be_bytes, Bytes, 0, N).


be_bytes(Byte, Acc, Next) :-
    Next is 256 * Acc + Byte.


write_words_with_hash(Stream, Words) :-
    length(Words, Length),
    write_u32(Stream, Length),
    append(Words, Bytes),
    maplist(write_u8(Stream), Bytes),
    sha_hash(Bytes, Hash, [algorithm(sha256), encoding(octet)]),
    maplist(write_u8(Stream), Hash).


write_u8(Stream, N) :-
    format(Stream, "~|~`0t~16r~2+", [N]).


write_u16(Stream, N) :-
    format(Stream, "~|~`0t~16r~4+", [N]).


write_u32(Stream, N) :-
    format(Stream, "~|~`0t~16r~8+", [N]).
