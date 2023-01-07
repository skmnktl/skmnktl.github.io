<template>
    <div class="aboutme">
      <span id="aboutmeimage">
        <img class="about" src="/winding_zion_road.jpg" />
        <figcaption>Angel’s Landing, Zion National Park</figcaption>
      </span>
      <span></span>
      <span id="aboutmedesc">
        <br />
        <p>
          I often write to learn. I expect this site to be a repository of all the things I'm learning about:
          <ul>
            <li class="topic">Carnatic Music</li>
            <li class="topic">Photography</li>
            <li class="topic">Sanskrit</li>
            <li class="topic">Math</li>
            <li class="topic">Computer Science</li>
          </ul>
        </p>
      </span>
    </div>
</template>

<script>
definePageMeta({
  layout: "default",
});
</script>

<style scoped>

ul{
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  list-style: none;
  margin: 0;
  padding: 0;
}

.topic {
  box-sizing: border-box;
  font-family: "Alegreya Sans", sans-serif;
  font-weight: 400;
  color: #000000;

}


.about{
    text-align: left;
    width: 350px;
    height: 350px;
    border-radius: 50px;
    padding: 30px;
}
  
@media only screen and (max-width: 600px) {
.about{
    align-content: left;
    width: 99%;
    height: 99%;
    border-radius: 50px;
    padding: 30px;
    }
}

p {}

div {
  font-family: "Alegreya Sans", serif;
}

.aboutme {
  font-family: "Alegreya Sans", serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #000000;
  display: inline-grid;
  grid-template-columns: 30% 5% 65%;
}

h3 {
  font-family: "Alegreya Sans SC", sans-serif;
  text-align: center;
}

p {
  font-family: "Alegreya Sans", sans-serif;
}


#aboutmedesc {
  text-align: left;
}
</style>


