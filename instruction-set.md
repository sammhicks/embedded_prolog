# Instruction Set

| instruction    |   B4 |   B3 |   B2 |   B1 |   B4 |   B3 |   B2 |   B1 | ... |
| -------------- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| put variable   | `00` | `Ai` | `00` | `Xn` |
| put variable   | `01` | `Ai` | `00` | `Yn` |
| put value      | `02` | `Ai` | `00` | `Xn` |
| put value      | `03` | `Ai` | `00` | `Yn` |
| put structure  | `04` | `Ai` |  `f` |  `n` |
| put structure  | `05` | `Ai` | `00` |  `n` | `00` | `00` | `f`1 | `f`0 |
| put list       | `06` | `Ai` | `00` | `00` |
| put constant   | `07` | `Ai` | `c`1 | `c`0 |
| put integer    | `08` | `Ai` | `i`1 | `i`0 |
| put integer    | `09` | `Ai` |  `S` |  `N` | `i`3 | `i`2 | `i`1 | `i`0 | ... |
| put void       | `0a` | `Ai` | `00` | `00` |
| get variable   | `10` | `Ai` | `00` | `Xn` |
| get variable   | `11` | `Ai` | `00` | `Yn` |
| get value      | `12` | `Ai` | `00` | `Xn` |
| get value      | `13` | `Ai` | `00` | `Yn` |
| get structure  | `14` | `Ai` |  `f` |  `n` |
| get structure  | `15` | `Ai` | `00` |  `n` | `00` | `00` | `f`1 | `f`0 |
| get list       | `16` | `Ai` | `00` | `00` |
| get constant   | `17` | `Ai` | `c`1 | `c`0 |
| get integer    | `18` | `Ai` | `i`1 | `i`0 |
| get integer    | `19` | `Ai` |  `S` |  `N` | `i`3 | `i`2 | `i`1 | `i`0 | ... |
| set variable   | `20` | `00` | `00` | `Xn` |
| set variable   | `21` | `00` | `00` | `Yn` |
| set value      | `22` | `00` | `00` | `Xn` |
| set value      | `23` | `00` | `00` | `Yn` |
| set constant   | `27` | `00` | `c`1 | `c`0 |
| set integer    | `28` | `00` | `i`1 | `i`0 |
| get integer    | `29` | `00` |  `S` |  `N` | `i`3 | `i`2 | `i`1 | `i`0 | ... |
| set void       | `2a` | `00` | `00` |  `n` |
| unify variable | `30` | `00` | `00` | `Xn` |
| unify variable | `31` | `00` | `00` | `Yn` |
| unify value    | `32` | `00` | `00` | `Xn` |
| unify value    | `33` | `00` | `00` | `Yn` |
| unify constant | `37` | `00` | `c`1 | `c`0 |
| unify integer  | `38` | `00` | `i`1 | `i`0 |
| unify integer  | `39` | `00` |  `S` |  `N` | `i`3 | `i`2 | `i`1 | `i`0 | ... |
| unify void     | `3a` | `00` | `00` |  `n` |
| allocate       | `40` | `00` | `00` |  `N` |
| trim           | `41` | `00` | `00` |  `N` |
| deallocate     | `42` | `00` | `00` | `00` |
| call           | `43` |  `N` | `P`1 | `P`0 |
| execute        | `44` |  `N` | `P`1 | `P`0 |
| proceed        | `45` | `00` | `00` | `00` |
| try me else    | `50` | `00` | `P`1 | `P`0 |
| retry me else  | `51` | `00` | `P`1 | `P`0 |
| trust me       | `52` | `00` | `00` | `00` |
| neck cut       | `53` | `00` | `00` | `00` |
| get level      | `54` | `00` | `00` | `Yn` |
| cut            | `55` | `00` | `00` | `Yn` |
| >              | `60` | `00` | `00` | `00` |
| <              | `61` | `00` | `00` | `00` |
| =<             | `62` | `00` | `00` | `00` |
| >=             | `63` | `00` | `00` | `00` |
| =\=            | `64` | `00` | `00` | `00` |
| =:=            | `65` | `00` | `00` | `00` |
| is             | `66` | `00` | `00` | `00` |
| true           | `70` | `00` | `00` | `00` |
| fail           | `71` | `00` | `00` | `00` |
| =              | `72` | `00` | `00` | `00` |
| system call    | `80` | `00` | `00` |  `I` |
