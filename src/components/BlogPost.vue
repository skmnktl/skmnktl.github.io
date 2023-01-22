<script setup lang="ts">
defineProps<{
  title: string,
  date: string,
  description: string,
  link: string,
}>()
</script>

<template>
  <br>
  <div class="blog-post-stub">
    <div class="title">{{title}}</div>
    <div class="date">{{date}}</div>
    <div class="desc">{{description}}</div>
    <div class="read-link"><RouterLink :to="link" style="color: #0078e9">Read →</RouterLink></div>
  </div>
  <br>
  <div>
    <RouterView />
  </div>
</template>

<style scoped>
  .blog-post-stub{
    text-align: left;
    width: 40%;
  }
  .title{
    display: grid;
    font-weight: bold;
    font-size: 1em;
  }
  .date{
    font-style: italic;
  }
  .desc{
    font-size: 0.75em;
  }

  .read-link{
    color: #0078e9 !important;
    font-size: 0.75em;
  }

  a:hover::after{
    content: " " !important;
  }
</style>
