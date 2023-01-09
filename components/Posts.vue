<template>
  <div class="postsSidebar">
    <span id="left">
      <br>
      <h1> Posts </h1>
      <h2> General Topics </h2>
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
    padding-left: 5px;
    font-weight: bold;
    font-size: 1.2em; 
}
h2 {
    padding-left: 10px;
    padding-right: 10px;
    margin-top: 0px;
    margin-bottom: 0px;
    font-weight: bold;
    font-size: 1em; 

}
#left {
  vertical-align: top;
  text-align: left;
}
.postsSidebar {
  width: 100%;
  text-align: left;
  vertical-align: top;
  display: grid;
  grid-template-columns: 30% 70%;
  font-family: 'Alegreya Sans';
}
</style>
