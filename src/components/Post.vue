<template>
  <span class="currentPost">
  <span class="leftSpacerPost"></span>
  <span class="postContent">
  <span class="backLink"><RouterLink to="/blog" style="color: #0078e9"> ← Back </RouterLink></span>

  <component :is="inputComponent"></component>

  </span>
  <span class="rightSpacerPost"></span>
  </span>
</template>

<script>
export default {
  props: {
    inputComponent: Object
  },
}
</script>

<style>
.currentPost{
  display: grid;
  grid-template-columns: 20% auto 20%;
}

.backLink {
  color: #0078e9 !important;
}

.backLink a:hover::after{
  content: " " !important;
}
</style>