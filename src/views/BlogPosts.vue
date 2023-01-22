<template>
  <span class="blog-page">
    <div class="blog-title">
    </div>
    <div class="blog-posts">
    <BlogPost
      link="/posts/interview-prep/interviewing"
      title="Interview Framework"
      date="1/20/2023"
      description="A consulting style framework for technical interview preparation"
    />
    <BlogPost
      link="/posts/interview-prep/ml_interview"
      title="ML Topics"
      date="1/20/2023"
      description="A list of fundamental topics for review before ML interviews"
    />
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
  margin-top: 1em;
  margin-bottom: 1em;
}

.blog-page{
  display: grid;
  font-size: 2em;
  text-align: center;
}
</style>

