<template>
  <h1 id="interview-method">Interview Method</h1>
  <p>The idea is that if I build up instincts for how to interview, the
    limited computing resources I have can be directed towards the simple
    act of problem-solving and the occasional joke to keep my interviewer
    entertained.</p>
  <p>The central idea is to do less simultaneously. The framework below
    helps me with that, but I think any schema that has the same effect
    would work. This often works not just for interviews, but for all sorts
    of performance anxiety.</p>
  <p>The goals in a coding interview are three; we’re to write code that:
    1. works 2. is efficient, and 3. is readable.</p>
  <p>I’ve adapted <a
    href="https://math.berkeley.edu/~gmelvin/polya.pdf">Polya</a>’s
    problem-solving list for this task, but crucially, I mark the parts of
    the list that are meant to be spoken out loud. Figuring out when and
    what to say is the worst of it. It’s easy to get lost in crafting the
    solution, and in writing the code. But these acts in themselves should
    be cues to speak. And when you have a cue in your code (perhaps a for
    statement), that should trigger speech and tell you what to say.</p>
  <p>I annotate each part to speak for with a caret below.</p>
  <h2 id="understand">Understand</h2>
  <ol type="1">
    <li>Inputs ^</li>
    <li>Outputs ^</li>
    <li>Examples (No Edge Cases) ^</li>
  </ol>
  <h2 id="plan">Plan</h2>
  <ol type="1">
    <li>Data Structure
      <ol type="1">
        <li>List, Linked List, HashMap, and Graph are the most common.</li>
        <li>Choose a structure. ^</li>
      </ol></li>
    <li>Sketch Out Rough Algorithm ^</li>
  </ol>
  <h2 id="write-code">Write Code</h2>
  <ol type="1">
    <li>Do we need to use OOP? ^</li>
    <li>Organizational Structures</li>
    <li>Complexity - Space vs. Time</li>
  </ol>
  <p>      Discussion of complexity is often implicitly about time. But
    that need not be the case. It’s good to ask or at least mention the
    assumption explicitly.</p>
  <h3 id="framework-by-structure">Framework by Structure</h3>
  <p>Speak briefly when writing each of the following structures. Writing
    any of the following structures should be a cue to speak.</p>
  <ol type="1">
    <li>Classes
      <ol type="1">
        <li>Methods ^</li>
        <li>Data ^</li>
      </ol></li>
    <li>Functions
      <ol type="1">
        <li>Parameters ^</li>
        <li>Return Types ^</li>
      </ol></li>
    <li>Loops
      <ol type="1">
        <li>What does a single iteration do? ^</li>
        <li>What are the termination conditions? ^</li>
      </ol></li>
    <li>Recursion
      <ol type="1">
        <li>What are the base cases? ^</li>
        <li>What are the termination conditions besides the base cases? ^</li>
        <li>Recursion is loop-based or function-based. The appropriate bullets
          for discussion in #2 and #3 apply.</li>
      </ol></li>
  </ol>
  <h2 id="test">Test</h2>
  <ol type="1">
    <li>Run Example Cases</li>
    <li>Edge Cases
      <ol type="1">
        <li>List: <span class="math inline">∅</span>, singleton list, list of
          two, etc.</li>
        <li>Linked List: Empty Root, etc.</li>
        <li>HashMap</li>
        <li>Graph: Empty Graph, Leaf (Leaf Node Type vs. None)
          <ol type="1">
            <li>Tree</li>
            <li>Binary Tree</li>
          </ol></li>
      </ol></li>
  </ol>
  <p>Certain data structures have common edge cases. A few are listed
    above, but spending time thinking of these is useful.</p>
  <h2 id="reflection">Reflection</h2>
  <ol type="1">
    <li>What could we have done to improve
      <ol type="1">
        <li>efficiency</li>
        <li>readability, and</li>
        <li>comments?</li>
      </ol></li>
    <li>Ask for feedback. Not all of it is useful, but sometimes, it can be
      gold.</li>
  </ol>
</template>