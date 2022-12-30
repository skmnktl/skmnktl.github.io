<template>
  <div class="posts">
    <span id="left">
      <Button id="interview" 
        title="Technical Interview Framework" 
        @click="changeDisplayedContent('interviewing')" />
      <Button id="ml_interview" 
        title="Topics for ML Interview Preparation" 
        @click="changeDisplayedContent('ml_interview')" />
    </span>
      <span id="right">
              <ContentDoc :path="contentPath">
              <template #not-found>
              </template>
              </ContentDoc>
      </span>
  </div>
</template>

<script>

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
.left {
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
