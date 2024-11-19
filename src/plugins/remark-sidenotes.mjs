import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkRehype from 'remark-rehype';
import rehypeRaw from 'rehype-raw';
import rehypeStringify from 'rehype-stringify';
import { visit } from 'unist-util-visit';

function markdownToHtml(markdown, index) {
  return unified()
    .use(remarkParse)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw) 
    .use(rehypeStringify)
    .processSync(`<span class="font-xxs align-super font-italic">[${index}] </span>`+markdown)
    .toString();
}

export default function remarkSidenote() {
  return (tree, file) => {
    let sidenoteCounter = 0;
    const markdownContent = file.value || '';
    const footnoteRegex = /\^\{(.+?)\}/g;
    const replacements = {};
    
    const updatedMarkdown = markdownContent.replace(footnoteRegex, (_, sidenoteText) => {
      const placeholder = `%%SIDENOTE_${sidenoteCounter}%%`;
      replacements[placeholder] = sidenoteText;
      sidenoteCounter++;
      return placeholder;
    });

    const newTree = unified()
      .use(remarkParse)
      .use(remarkRehype, { allowDangerousHtml: true })
      .use(rehypeRaw)
      .parse(updatedMarkdown);

    visit(newTree, 'text', (node) => {
      if (node.value && node.value.includes('%%SIDENOTE_')) {
        Object.entries(replacements).forEach(([placeholder, sidenoteText], index) => {
          if (node.value.includes(placeholder)) {
            node.type = 'html';
            node.value = node.value.replace(placeholder, `
              <span class="relative sidenote-anchor align-super">
                <span class="sidenote-number" data-sidenote-id="${index + 1}">${index + 1}</span>
              </span>`)
              +`
              <span id="sidenote-placeholder-${index + 1}" class="sidenote">
                <span>${markdownToHtml(sidenoteText, index+1)}</span>
              </span>
            `;
          }
        });
      }
    });

    tree.children = newTree.children;
  };
}
