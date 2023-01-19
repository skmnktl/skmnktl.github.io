<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import HelloWorld from './components/HelloWorld.vue'
</script>

<template>
  <header>
    <div class="wrapper">
      <nav>
        <span class="toolbarLeft">srīnivāsa kaśyap munukutla</span>
        <span class="toolbarSpacer"></span>
        <span class="toolbarRight">
          <RouterLink to="/">home</RouterLink>
          <RouterLink to="/resume">resume</RouterLink>
          <RouterLink to="/about">about</RouterLink>
        </span>
      </nav>
    </div>
  </header>

  <RouterView />
</template>

<style scoped>
.wrapper nav{
  display: grid;
  grid-template-columns: 30% auto 30%;
  font-size: 1.5em;
}
.toolbarLeft{
  text-align: left;
}

.toolbarRight{
  text-align: right;
}

@media only screen and (max-width: 600px) {
  .wrapper nav{
    font-size: 1em;
  }
}

</style>
