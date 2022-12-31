<template>
<button class="button" style="vertical-align: middle">
  <font-awesome-icon :icon="['fa-solid','fa-pen-nib']" /> {{ title }}
</button>
</template>

<script>

export default {
  props: {
    id: String,
    title: String
  }
}

</script>


<style scoped>
.button {
  display: inline-block;
  border-radius: 4px;
  border: none;
  text-align: center;
  font-size: 0.75em;
  transition: all 0.5s;
  cursor: pointer;
  margin: 5px;
  padding-right: 10px;
  padding-left: 10px;
  background-color: transparent;
  color: blue;
  font-family: "Alegreya Sans SC", sans-serif;
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
