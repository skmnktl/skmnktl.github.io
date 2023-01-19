import { library, config } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
//import { faArrowRight as fasArrowRight, faArrowLeft as fasArrowLeft } from '@fortawesome/free-solid-svg-icons'
import { faLinkedin as fabLinkedin } from '@fortawesome/free-brands-svg-icons'

config.autoAddCss = false

library.add(
    fabLinkedin,
   // fasArrowLeft,
   // fasArrowRight
)

export default defineNuxtPlugin((nuxtApp) => {
  nuxtApp.vueApp.component('font-awesome-icon', FontAwesomeIcon)
})