<template>
  <div id="aboutme">
    <span id="left"></span>
    <span id="middle"> 
      <span class="container">
          <span class="aboutitems" id="aboutmeimage">
            <img class="about-img" src="/winding_zion_road.jpg" />
            <figcaption><strong>Angel’s Landing, Zion National Park</strong></figcaption>
          </span>
        <span class="aboutitems" id="desc">
                <p class="desc">
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
        
      </span>
    </span>
    <span id="right"></span>
  </div>
</template>

<script>
definePageMeta({
  layout: "default",
});
</script>

<style scoped>
#aboutme {
  display: inline-grid;
  grid-template-columns: 30% 40% 30%;
  width: 100%;
}

@media only screen and (max-width: 600px) {
  #aboutme {
  display: inline-grid;
  grid-template-columns: 10% 80% 10%;
  width: 100%;
  }
}

#left {
}
#middle {
  align-content: center;
  text-align: center;
}
#desc{
  text-align: left;
  align-content: left;
}
#right {
}

ul {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  list-style: none;
  margin: 0;
  padding-left: 20px;
}

li:before{
  content: "\2014";
  margin-left: 0em;
  margin: 0.5em;
}

.about-img {
  width: 100%;
  border-radius: 50px;
  text-align: center;
  max-width: 480px;
  align-content: center;
  display: block;
  margin-left: auto;
  margin-right: auto;
}

figcaption {
  text-align: center;
}
</style>