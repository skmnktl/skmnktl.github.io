---
title: 'Code Quality'
description: Summaries of other people's thoughts on code quality reduced to a checklist.
tags:
  - software_engineering
date: 2024-08-23
---

# Writing Code^{None of this is original thought; it’s just a summary of other sources on code quality.}

## Goals
Code should be _performant_, _easy to read_, _easy to change_, and _easy to test_.

## Process

### Before PR

*  What are you even building?  
*  Design
  *  Clarify Scope
  *  Define PR Chunks
    * PRs should cascade; test the chunks independently and test them all together too before merging into `main`.
*  Review Design (if appropriate)

### PR
*  Code
*  Write Tests
*  Refactor and Tidy for Code Quality (see below)
*  Test locally
*  Tag with RC
*  Deploy to Stage and Test
*  `git checkout main; git pull; git checkout feature-branch; git merge main`
*  Perform Final Tests on RC after main merge
*  Submit PR
*  Address PR Comments
*  Do NOT Merge

### Deployment^{After PR Approval}
*  Send Release Notification
*  Release Meeting
*  Merge main back in.^{`git checkout main; git pull; git checkout feature-branch; git merge main`}
  *  Merge Conflicts: Do we need to review again?
*  Merge, Immediately tag Release and Deploy

## Design & Style Philosophies

### Easy to Read^{_Tidy First?_ by Kent Beck}
1. Avoid nested or chained conditionals with *guard clauses* where they increase simplicity.
2. Remove dead code carefully.
3. Refactor *symmetric code* to use common patterns or abstractions; conform to standard patterns used across the team.
4. Write a cleaner, simpler *new interface* for an existing implementation— keep the old implementation till backward compatibility is unnecessary. 
5. Order 
   -  Ensure that code reads logically from top to bottom. Reorder code blocks to follow the natural order of execution.
   -  Code should cohere at each level: 
        * Code Blocks, Functions/Methods^{A *function* associated with a specific object is a *method*.}, Classes/Modules
        * Variables, Constants, Parameters
        * Code Blocks
          * Chunk complex statements into logical blocks; use whitespace appropriately.
          * Introduce intermediate values to reduce cognitive load. 
6. Declare and initialize variables in one place; otherwise, refactor code.
7. Explaining: Variables, Constants, and Parameters
   * Use explanatory names, and where names are insufficient, use comments.
   * Comments should be about why rather than what. If a comment about 
8. Extract helper functions or methods; reduce repetition. 
9. It can help to dump all logic into a single place before refactoring. 
10. Add comments to explain why rather than what; if you must explain what, it bears thinking about how you might refactor the code.
11. Remove unnecessary comments; make sure to update existing ones. 

### Easy to Test^{[Misko Hevery](https://testing.googleblog.com/2008/08/by-miko-hevery-so-you-decided-to.html)}

> There are no tricks to writing tests; there are only tricks to writing testable code. -- Misko Hevery

1. Separate object construction and the application's work.
2. Ask for things, don't look for things ^{Dependency Injection / Law of Demeter}
3. Avoid 
   - Work within Constructors
   - Global State
   - Singletons^{This is just global state in disguise.}
   - Static Methods
   - Mixing Concerns^{enforce single responsibilty}
4. Favor 
   - Composition over Inheritance
   - Polymorphism over Conditionals
5. Separate data from the work.

### Easy To Change^{Jon Ousterhout's *A Philosophy of Software Design*, perhaps?} 

1. Complexity is anything that makes a system hard to understand or change.
  -   


### Easy to Use 

