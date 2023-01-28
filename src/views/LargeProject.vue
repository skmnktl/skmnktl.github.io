<template>
  <span class="blog-page">
    <div class="blog-title">
    </div>
    <div class="blog-posts">

    </div>
  </span>
  <span class="display-posts">
    <RouterView />
  </span>
</template>

<script setup>
import BlogPost from "@/components/BlogPost.vue"
</script>

<style scoped>
.blog-title{
  text-align: left;
  font-style: bold;
  margin-top: 0em;
  margin-bottom: 0em;
}

.blog-page{
  display: grid;
  font-size: 2em;
  text-align: center;
}
</style>

