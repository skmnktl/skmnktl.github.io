// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
    app: {
        head: {
            link: [{
                rel: 'stylesheet',
                href: 'https://fonts.googleapis.com/css2?family=Alegreya+SC:ital,wght@0,400;0,500;0,700;0,800;1,400;1,500;1,700;1,800&family=Alegreya+Sans+SC:ital,wght@0,100;0,300;0,400;0,500;0,700;0,800;1,100;1,300;1,400;1,500;1,700;1,800&family=Alegreya+Sans:ital,wght@0,100;0,300;0,400;0,500;0,700;1,100;1,300;1,400;1,500;1,700&family=Alegreya:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500;1,600;1,700&display=swap'
            }],
            script: [{
                src: 'https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js',
            }],
        },
        pageTransition: { name: 'page', mode: 'out-in' },
        layoutTransition: { name: 'slide', mode: 'out-in'},
    },
    css: [
        '~/assets/fonts/alegreya.css'
    ],
    modules: [
        '@nuxt/content'
    ],
    content: {
        documentDriven: true
    }
});