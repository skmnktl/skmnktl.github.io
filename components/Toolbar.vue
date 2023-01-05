<template>
  <div class="toolbar">
    <span id="name"> śrīnivāsa kaśyap munukutla </span>
    <span id="links">
      <button class="button" style="vertical-align: middle">
        <span
          ><NuxtLink to="/" align="right" style="text-decoration: none">
            about me
          </NuxtLink>
        </span>
      </button>
      <button class="button" style="vertical-align: middle">
        <span
          ><NuxtLink to="/resume" align="right" style="text-decoration: none">
            resume
          </NuxtLink>
        </span>
      </button>
      <button class="button" style="vertical-align: middle">
        <span
          ><NuxtLink
            to="/photography"
            align="right"
            style="text-decoration: none"
          >
            photography
          </NuxtLink>
        </span>
      </button>
      <button class="button" style="vertical-align: middle">
        <span
          ><NuxtLink
            to="https://skmnktl.github.io/blog"
            align="right"
            style="text-decoration: none"
          >
            blog
          </NuxtLink>
        </span>
      </button>
    </span>
  </div>
</template>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.toolbar {
  float: left;
  width: 100%;
  clear: none;
  display: grid;
  grid-template-columns: 50% 50%;
  font-family: "Alegreya Sans", sans-serif;
  color: black;
}
#name {
  float: left;
  font-size: 1em;
  align-content: center;
  font-weight: medium;
  font-family: "Alegreya Sans SC", sans-serif;
  vertical-align: bottom;
  margin: 5px;
  padding-right: 10px;
  padding-left: 10px;
}
#links {
  float: left;
  font-family: "Alegreya Sans", sans-serif;
  font-size: 1em;
  text-decoration: none;
  vertical-align: bottom;
  text-align: right;
  color: black;
}

.button {
  display: inline-block;
  border-radius: 4px;
  border: none;
  text-align: center;
  font-size: 1em;
  transition: all 0.5s;
  cursor: pointer;
  margin: 5px;
  padding-right: 10px;
  padding-left: 10px;
  background-color: transparent;
  color: black;
  font-family: "Alegreya Sans", sans-serif;
}

.button span {
  cursor: pointer;
  display: inline-block;
  position: relative;
  transition: 0.5s;
}

.button span:after {
  content: "\00bb";
  position: absolute;
  opacity: 0;
  top: 0;
  right: -20px;
  transition: 0.5s;
}

.button:hover span {
  padding-right: 20px;
}

.button:hover span:after {
  opacity: 1;
  right: 0;
}
</style>