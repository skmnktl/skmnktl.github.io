<template>
  <div class="posts">
    <span id="left">
      <br>
      <h1> Posts </h1>
      <Button id="interview" 
        title="Technical Interview Framework" 
        @click="changeDisplayedContent('/general/interviewing')" />
      <Button id="ml_interview" 
        title="Topics for ML Interview Preparation" 
        @click="changeDisplayedContent('/general/ml_interview')" />
    </span>
    <span id="right">
      <iframe id="postFrame" :src="contentPath" frameBorder="0">

      </iframe>
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
      this.contentPath = pathToContent;
    }
  }
}
</script>

<style scoped>
#postFrame{
  width: 100%;
  min-height: 1500px;
  height: auto;
}

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
