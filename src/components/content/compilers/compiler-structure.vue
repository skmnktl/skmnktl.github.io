<template>
<body>
<h1 id="compiler">Compiler</h1>
<p>A compiler takes a stream of tokens in a <code>src</code> language,
and it translates it to the <code>trg</code> language.</p>
<h1 id="phases">Phases</h1>
<pre class="mermaid"><code>    graph TD;
      src --&gt; Frontend;
      Frontend --&gt; Intermediate Representation;
      Intermediate Representation --&gt; Backend;
      Backend -&gt; trg;</code></pre>
<h2 id="frontend">Frontend</h2>
<p>The frontend consists of the following: 1. Scanning 2. Lexical
Analysis 3. Syntax Analysis</p>
</body>
</template>
<script src="//cdnjs.cloudflare.com/ajax/libs/html5shiv/3.7.3/html5shiv-printshiv.min.js">

</script>

