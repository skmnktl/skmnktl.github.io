# Interview Method
The idea is that if I build up instincts for how to interview, the limited computing resources I have can be directed towards the simple act of problem-solving and the occasional joke to keep my interviewer entertained. 

The central idea is to do less simultaneously. The framework below helps me with that, but I think any schema that has the same effect would work. This often works not just for interviews, but for all sorts of performance anxiety. 

The goals in a coding interview are three; we’re to write code that:
1. works
2. is efficient, and
3. is readable. 

I’ve adapted [Polya](https://math.berkeley.edu/~gmelvin/polya.pdf)’s problem-solving list for this task, but crucially, I mark the parts of the list that are meant to be spoken out loud. Figuring out when and what to say is the worst of it. It’s easy to get lost in crafting the solution, and in writing the code. But these acts in themselves should be cues to speak. And when you have a cue in your code (perhaps a for statement), that should trigger speech and tell you what to say. 

I annotate each part to speak for with a caret below. 

## Understand
1. Inputs ^
2. Outputs ^
3. Examples (No Edge Cases) ^ 

## Plan
1. Data Structure 
	1. List, Linked List, HashMap, and Graph are the most common.
	2. Choose a structure. ^
2. Sketch Out Rough Algorithm ^ 

## Write Code
1. Do we need to use OOP? ^ 
2. Organizational Structures
3. Complexity - Space vs. Time

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Discussion of complexity is often implicitly about time. But that need not be the case. It's good to ask or at least mention the assumption explicitly. 

### Framework by Structure

Speak briefly when writing each of the following structures. Writing any of the following structures should be a cue to speak.

1. Classes
    1. Methods ^
    2. Data ^ 
2. Functions
    1. Parameters ^
    2. Return Types ^
3. Loops
    1. What does a single iteration do? ^
    2. What are the termination conditions? ^
4. Recursion
    1. What are the base cases? ^
    2. What are the termination conditions besides the base cases? ^
    3. Recursion is loop-based or function-based. The appropriate bullets for discussion in #2 and #3 apply.   


## Test
1. Run Example Cases
2. Edge Cases
	1. List: $\emptyset$, singleton list, list of two, etc. 
	2. Linked List: Empty Root, etc.
	3. HashMap
	4. Graph: Empty Graph, Leaf (Leaf Node Type vs. None)
		1. Tree
		2. Binary Tree

Certain data structures have common edge cases. A few are listed above, but spending time thinking of these is useful.   

## Reflection
1. What could we have done to improve 
	1. efficiency
	2. readability, and
	3. comments?
2. Ask for feedback. Not all of it is useful, but sometimes, it can be gold. 
