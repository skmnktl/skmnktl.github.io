<template>
    <div class="aboutme">
      <span id="aboutmeimage">
        <img class="about" src="../static/winding_zion_road.jpg" />
      </span>
      <span id="aboutmedesc">
        <br />
        <h3>Angel’s Landing, Zion National Park</h3>
        <p>
          I remember little of going up this cliff; my memories are mostly of
          the time at the top. I should have taken more photos along the way.
          This blog is to my life what those photos would have been for my trip
          to Utah.
        </p>
      </span>
    </div>
</template>

<style scoped>
.about{
    text-align: left;
    width: 500px;
    height: 500px;
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

div {
  font-family: "Alegreya Sans", serif;
}

.aboutme {
  font-family: "Alegreya Sans", serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #000000;
}

h3 {
  font-family: "Alegreya Sans SC", sans-serif;
}

p {
  font-family: "Alegreya Sans", sans-serif;
}


#aboutmedesc {
  text-align: left;
}
</style>