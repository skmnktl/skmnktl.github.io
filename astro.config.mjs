import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkSidenote from './src/plugins/remark-sidenotes.mjs';

export default defineConfig({
  site: 'https://skmnktl.github.io', 
  base: '/', 
  integrations: [mdx()],
  markdown: {
    remarkPlugins: [remarkMath, remarkSidenote],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      theme: 'dracula',
      wrap: true
    }
  }
});
