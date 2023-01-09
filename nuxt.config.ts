// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
    target: 'static',
    ssr: true,
    app: {
        head: {
            link: [{
                rel: 'stylesheet',
                href: 'https://fonts.googleapis.com/css2?family=Alegreya+Sans+SC:ital,wght@0,100;0,300;0,400;0,500;0,700;0,800;1,100;1,300;1,400;1,500;1,700;1,800&family=Alegreya+Sans:ital,wght@0,100;0,300;0,400;0,500;0,700;1,100;1,300;1,400;1,500;1,700&display=swap'
            },
            {
                rel:'preconnect',
                href: 'https://fonts.googleapis.com'
            },
            {
                rel: 'preconnect',
                href: 'https://fonts.gstatic.com',
                crossorigin: true
            },
            {
                rel: 'stylesheet',
                href: 'https://cdn.jsdelivr.net/npm/katex@0.11.0/dist/katex.min.css'
            }],
            script: [{
                src: 'https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js',
            }],
        },
        pageTransition: { name: 'page', mode: 'out-in' },
        layoutTransition: { name: 'slide', mode: 'out-in'},
    },
    css: [
        '@fortawesome/fontawesome-svg-core/styles.css',
        '~/assets/css/main.css'
    ],
    modules: [
        '@nuxt/content'
    ],
    nitro: {
        prerender: {
            crawlLinks: false,
            routes: ['/','/resume','/photography','/general/interviewing','/general/ml_interview']
        }
    },
    experimental :{
        payloadExtraction: false
    },
    markdown: {
      remarkPlugins: [
        'remark-math'
      ],
      rehypePlugins: [
        'rehype-katex'
      ]
    },
    postcss: {
    plugins: {
      tailwindcss: {},
      autoprefixer: {},
    },
  },
});
