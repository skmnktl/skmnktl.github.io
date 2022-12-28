<template>
  <div id="photos">
    <span class="container" key="1">
      <figure class="figurefx pushup">
        <img src="~/assets/img/1.jpg" class="image" />
        <figcaption>Zion National Park, Utah</figcaption>
      </figure>
    </span>
    <span class="container" key="2">
      <figure class="figurefx pushup">
        <img src="~/assets/img/2.jpg" class="image" />
        <figcaption>Great Smoky Mountains National Park</figcaption>
      </figure>
    </span>
    <span class="container" key="3">
      <figure class="figurefx pushup">
        <img src="~/assets/img/3.jpg" class="image" />
        <figcaption>śarasvati vīṇa</figcaption>
      </figure>
    </span>
    <span class="container" key="4">
      <figure class="figurefx pushup">
        <img src="~/assets/img/4.jpg" class="image" />
        <figcaption>Lake Tahoe</figcaption>
      </figure>
    </span>
    <span class="container" key="5">
      <figure class="figurefx pushup">
        <img src="~/assets/img/5.jpg" class="image" />
        <figcaption>Chantilly, VA</figcaption>
      </figure>
    </span>
    <span class="container" key="6">
      <figure class="figurefx pushup">
        <img src="~/assets/img/6.jpg" class="image" />
        <figcaption>Heraklion, Greece</figcaption>
      </figure>
    </span>
  </div>
</template>

<style scoped>
div {
  font-family: "Alegreya Sans", serif;
}

.image {
  width: 500px;
  height: auto;
}

@media only screen and (max-width: 600px) {
.image{
    width: 99%;
    height: 99%;
    }
}

figure.figurefx {
  width: 600px;
  margin: 30px auto;
  padding: 0;
  display: block;
  position: relative;
  overflow: hidden; /* hide overflowing elements by default */
}

figure.figurefx figcaption {
  position: absolute;
  display: block;
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
  text-align: center;
  padding: 15px;
  z-index: 100;
  width: 100%;
  max-height: 100%;
  overflow: hidden;
  top: 50%;
  left: 0;
  -moz-transform: translate3d(
    -100%,
    -50%,
    0
  ); /* position caption outside layout horizontally and centered vertically */
  -webkit-transform: translate3d(-100%, -50%, 0);
  transform: translate3d(-100%, -50%, 0);
  opacity: 0;
  -moz-transition: all 0.5s;
  -webkit-transition: all 0.5s;
  transition: all 0.5s;
}

figure.figurefx figcaption a {
  text-decoration: none;
}

/*** Push Up effect ****/
figure.pushup img {
  -moz-transform: translate3d(0, 0, 0);
  -webkit-transform: translate3d(0, 0, 0);
  transform: translate3d(0, 0, 0);
  -moz-transition: all 0.5s;
  -webkit-transition: all 0.5s;
  transition: all 0.5s;
}

figure.pushup figcaption {
  top: 100%;
  opacity: 1;
  -moz-transform: translate3d(0, 0, 0);
  -webkit-transform: translate3d(0, 0, 0);
  transform: translate3d(0, 0, 0);
}

figure.pushup:hover img {
  -moz-transform: translate3d(
    0,
    -40px,
    0
  ); /* move image up vertically a bit for parallax effect */
  -webkit-transform: translate3d(0, -40px, 0);
  transform: translate3d(0, -40px, 0);
  opacity: 0.5;
}

figure.pushup:hover figcaption {
  -moz-transform: translate3d(
    0,
    -100%,
    0
  ); /* move caption up vertically to fully reveal caption */
  -webkit-transform: translate3d(0, -100%, 0);
  transform: translate3d(0, -100%, 0);
  -moz-transition: all 0.5s;
  -webkit-transition: all 0.5s;
  transition: all 0.5s;
  border-radius: 10px;
}

#photos {
  font-family: "Alegreya Sans", sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #000000;
}
</style>