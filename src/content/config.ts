import { defineCollection, z } from 'astro:content';

const posts = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.date(),
    draft: z.boolean().default(false),
    // Optional cover image for blog posts
    coverImage: z.object({
      src: z.string(),
      alt: z.string()
    }).optional()
  })
});

export const collections = {
  posts: posts,
};
