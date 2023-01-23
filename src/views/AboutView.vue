<template>
  <div class="about">
    <span class="aboutSpacerLeft"></span>
    <span class="aboutMiddle">
    <figure id="aboutFig">
      <img id="aboutImage" src="@/assets/winding_zion_road.jpg" />
      <figcaption><span id="caption">Angel's Landing at Utah's Zion National Park</span></figcaption>
    </figure>
    </span>
    <span class="aboutSpacerRight"></span>
  </div>
</template>

<style scoped>



img{
  width: 30%;
  max-width: 70%;
  border-radius: 50px;
}
figure {
  text-align: center;
  margin: 10px auto;
}
figcaption{
  font-size: 1.25em;
}

@media only screen and (max-width: 600px) {
  .about{
    display: flex;
    font-size: 1em;
  }

  #caption{
    font-size: 0.8em;
  }

  img{
    width: 90%;
    max-width: 95%;
    border-radius: 50px;
  }
}
</style>
