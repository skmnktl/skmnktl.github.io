<template>
  <div>
    <NuxtLayout>
    <NuxtPage />
    </NuxtLayout>
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

<style>
.page-enter-active,
.page-leave-active {
  transition: all 0.5s;
}
.page-enter-from,
.page-leave-to {
  opacity: 0;
  filter: grayscale(1);
}
</style>
