foo(A, B, A, D, D) :-
    a(A),
    b(B).

a(a).
b(b).


bar(foo(A, x), A, A).

p(_, g(_), f(_, _, _)).
