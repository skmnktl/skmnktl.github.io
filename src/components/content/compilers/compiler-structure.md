# Compiler

A compiler takes a stream of tokens in a `src` language, and it translates it to the `trg` language. 
 
# Phases
```mermaid
    graph TD;
      src --> Frontend;
      Frontend --> Intermediate Representation;
      Intermediate Representation --> Backend;
      Backend -> trg;
```
## Frontend
The frontend consists of the following:
1. Scanning
2. Lexical Analysis
3. Syntax Analysis
