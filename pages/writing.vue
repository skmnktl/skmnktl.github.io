<template>
    <div class="posts">
    <span id="left">
      <br>
      <h1> Posts </h1>
      <Button id="interview" 
        title="Technical Interview Framework" 
        @click="changeDisplayedContent('interviewing')" />
      <Button id="ml_interview" 
        title="Topics for ML Interview Preparation" 
        @click="changeDisplayedContent('ml_interview')" />
    </span>
    <span id="right-post">
      <ContentDoc :path="contentPath">
        <template #not-found>
          <br>
          <br>
          Click to display a post. 
        </template>
      </ContentDoc>
      </span>
  </div>
</template>

<script scoped>
export default {
  data() {
    return {
      contentPath: ""
    }
  },
  methods: {
    changeDisplayedContent(pathToContent) {
      this.contentPath = "/" + pathToContent;
    }
  }
}
</script>

<style scoped>
h1 {
    margin:0;
}

#left {
  vertical-align: top;
  text-align: left;
}
.posts {
  width: 100%;
  text-align: left;
  vertical-align: top;
  display: grid;
  grid-template-columns: 30% 70%;
}
</style>
