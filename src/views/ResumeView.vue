<template>

  <div class="root-resume">
    <span class="spacerLeft"></span>
    <span class="resumeContent" id="pdfThis">
    <section class="main-heading">
    <h1 class="my-name">Kasyap Munukutla</h1>
    <h1 class="my-linkedin"><a href="http://linkedin.com/in/kasyapmunukutla"><font-awesome-icon icon="fab fa-linkedin-in" /></a></h1>
    </section>
      <section id="objective">
          My eventual goal is a move to quant-finance or fintech industries; I bring several years of experience leading analytics teams as well as research engineering in NLP and building ML Data Pipelines and Infrastructure. Most recently, I've been working in Backend Development to round out my core engineering toolkit.
      </section>
    <section id="work-experience">
    <h1 class="section-title">Work Experience</h1>

    <section class="employer">
    <h2 class="name"><a href="http://samsung.com/">Samsung, Inc.</a></h2>
    <p class="job-title">Sr. Backend Engineer | Jul 2022 - Current</p>
    <p class="job-description"><strong>Built and deployed microservices for Samsung TV+ streaming service</strong></p>
    <ul>
      <li>Developed and rebuilt several APIs integrating various AWS services into Samsung's channel service to improve
        code modularity and cleanliness
      </li>
    </ul>
    <span class="relevant-skills"><em>Relevant Skills</em> Python, Go, AWS, Docker, Github Actions, Microservice Design &amp; Implementation </span>
    </section>

    <section class="employer">
    <h2 class="name">UChicago, CS Department</h2>
    <p class="job-title">Research Engineer, NLP for Prof. Chenhao Tan | Mar 2022 - Aug 2022</p>
    <p class="job-description"><strong>Built model predicting whether evoking a given emotion is persuasive using Reddit's ChangeMyView
      data</strong></p>
    <ul>
      <li><u>Data</u>: OP asks for their view to be changed and awards a point to each convincing reply; each thread is a tree
        with children being responses to parent posts
      </li>
      <li><u>Pre-processing</u>: posts are tagged with the corresponding label: convincing or not then tokenized using nltk's
        tokenizer
      </li>
      <li><u>Structure</u>: posts are redacted to reflect only vocabulary for a specific emotion, e.g. fear; word2vec
        embedding-based negative sampling technique used since 99.5% of data is negatively labeled
      </li>
      <li><u>Model</u>: BERT-Transformer model implemented using PyTorch, PyTorch Lightning and HuggingFace Transformer</li>
      <li><u>Results</u>: fear, belligerence, and qualification are all highly predictive (between 75% to 65% compared to
        baseline of 60%); bluster and confidence are not
      </li>
    </ul>
    <span class="relevant-skills"><em>Relevant Skills</em> PyTorch, PyTorch Lightning, NLP (NLTK, HuggingFace Transformers, RNNs, Tokenization, Word Embeddings), NumPy, Go </span>
    </section>

    <section class="employer">
    <h2 class="name">UChicago, Booth School of Business</h2>
    <p class="job-title">Research Engineer, Finance for Prof. Jessica Jeffers | Jan 2020 - Jul 2021 </p>
    <p class="job-description"><strong>Conducted research on antitrust policy and economic growth using congressional transcripts, antitrust
      case records, and business locations</strong></p>
    <ul>
      <li>Produced 5 datasets linking Senate speeches and speakers to campaign contributions and businesses within
        Senators’ constituencies; instituted processes reducing data ingestion time from months to days making datasets
        useful for years to come
      </li>
      <li>built a datastore and ETL pipeline replacing data storage on drives with a well-organized data repository
        including data dictionaries and entity diagrams; allowed ramp-up of new researchers and increased research speed
        by 5x
      </li>
    </ul>
      <span class="relevant-skills"><em>Relevant Skills</em> Python,SQL, NLP (Entity Recognition,Topic Extraction, Speaker Tagging), GeoJSON Shape Files, Google BigQuery, Data Engineering</span>
    </section>

    <section class="employer">
    <h2 class="name"><a href="http://quinstreet.com/">Quinstreet, Inc.</a></h2>
    <p class="job-title">Sr. Manager, Analytics | Oct 2019 - Aug 2020 </p>
    <p class="job-description"><strong>Managed a team of four with oversight of financial and operational reporting as well as being the key
      strategic input for the email marketing team</strong></p>
    <ul>
      <li>implemented logistic regression and decision tree models to score and filter incoming third-party data sources
        to better target marketing recipients yielding 40% increase in revenues
      </li>
      <li>automated our processes with Python and HiveQL which resulted in reduced time reporting and manual tasks by
        60%
      </li>
      <li>transformed a team reporting in Excel into a team working on prediction and automation using SQL, Hadoop,
        Tableau, and Python thereby allowing decision-makers daily data refreshes rather than monthly
      </li>
    </ul>
    </section>

    <section class="employer">
    <h2 class="name">Stanford University, Graduate School of Business</h2>
    <p class="job-title">Research Engineer, Computational Sociology for Prof. Amir Goldberg | Jul 2016 - Mar 2018</p>
    <p class="job-description"><strong>Conducted sociological research on enculturation of employees within organizations using 10M+ records of
      email data</strong></p>
    <ul>
      <li>produced structured data from variously encoded emails (Gmail, Hotmail, etc.), redacted sensitive information
        and tokenized data to produce 5 different compact and useable datasets
      </li>
      <li>built LIWC-based bag-of-words model pipeline to measure linguistic similarity between individuals and groups
        producing data used by 4+ researcher sub-teams
      </li>
      <li>created modular and performant data-processing pipeline; improved ETL times from several months to weeks
        thereby convincing 3+ partners to share sensitive data with the research team
      </li>
    </ul>
    <span class="relevant-skills"><em>Relevant Skills</em> Python, Java, NLP (Spacy, Text Parsing, Speaker Taggin, Tokenization, Word Embeddings), NumPy, Data Engineering, EML Files </span>
    </section>
    </section>

    <section class="employer">
    <h2 class="name"><a href="http://sojern.com/">Sojern, Inc.</a></h2>
    <p class="job-title">Sr. Data Analyst (Lead) | Jan 2016 - Oct 2019</p>
    <p class="job-description"><strong>Managed junior colleagues on various analytical and large-scale automation projects including:</strong>
    </p>
    <ul>
      <li>parsing and loading 1K+ client performance reports going from weekly to hourly data refreshes to enable the
        development of a flagship ML-based campaign optimization product
      </li>
      <li>managed analytical projects for CEO and exec-team thereby contributing to a successful $100M+ fundraising
        round
      </li>
      <li>led preparation for Board of Directors meetings on behalf of CEO-staff: maintained official metrics, explained
        KPI trends, and performed ad-hoc data analysis
      </li>
      <li>predicted recurrent revenue and customer attrition using regression and decision trees in a stacked model in preparation for successful funding round</li>
    </ul>
      <span class="relevant-skills"><em>Relevant Skills</em> Python, Tableau, SQL, Google BigQuery</span>
    </section>

    <section class="employer">
    <h2 class="name"><a href="http://cardinalhealth.com/">Cardinal Health</a></h2>
    <p class="job-title">Corporate Strategy, Operations Finance, FP&amp;A | Jul 2012 - Jan 2016</p>
      <ul>
      <li>Consultant, Corporate Strategy | Jun. 2014 - Jan 2016</li>
      <li>Sr. Analyst, Finance | Jul. 2012 - Jun. 2014</li>
      </ul>
    </section>


    <section id="education">
    <h1 class="section-title">Education</h1>
    <section class="school">
    <h2 class="name"><a href="http://uchicago.edu/">University of Chicago</a></h2>
    <p class="degree-title">MS, Computer Science with specialization in Machine Learning | Jun 2022 </p>
    </section>

    <section class="school">
    <h2 class="name"><a href="https://www.virginia.edu/">University of Virginia</a></h2>
    <p class="degree-title">BA, Economics | May 2012 </p>
    </section>

    </section>

    <section id="skills">
    <h1 class="section-title">Skills</h1>
    <ul class="skills-list">
      <li>Python (Numpy/PyTorch/Pandas/Flask/FastAPI)</li>
      <li>SQL</li>
      <li>C++</li>
      <li>C</li>
      <li>Rust</li>
      <li>Go</li>
      <li>Haskell</li>
      <li>Docker/Docker Compose</li>
      <li>Latex</li>
      <li>Web Dev (JS/HTML/CSS/VueJS/Nuxt)</li>
      <li>BigQuery</li>
      <li>AWS</li>
    </ul>
  </section>
      </span>
    <span class="spacerLeft">
      <button @click="exportToPDF" class="pdfGenToggle">pdf</button>
    </span>
  </div>
</template>
<script>
import html2pdf from "html2pdf.js"

export default {
  setup(){
    function exportToPDF(){
      var element = document.getElementById('pdfThis');
      var opt = {
        margin:       0.5,
        filename:     'myfile.pdf',
        image:        { type: 'jpeg', quality: 1 },
        html2canvas:  { scale: 4 },
        jsPDF:        { unit: 'in', format: 'a4', orientation: 'portrait' },
        pagebreak: { mode: 'avoid-all', before: '.page-break' }
      };
      html2pdf().set(opt).from(element).save();
    }
    return {
      exportToPDF
    }
  }
}
</script>

<style scoped>
#objective{
  font-style: italic;
  font-size: 1.25em;
  margin-bottom: 0.25em;
}
.pdfGenToggle{
  display: none;
}

.page-break{
  break-before: always;
}

.root-resume {
  display: grid;
  grid-template-columns: auto 60% auto;
}
.resumeContent{
  margin-bottom: 0em;
  margin-top: 0em;
  font-size: 24px;
}
.fa-linkedin-in{
  color: #0077b5;
}

.main-heading {
  margin-top: 0.75em;
  margin-bottom: 0.75em;
}

.my-name {
  display: inline;
  font-size: 2em;
}

.my-linkedin::before {
  content: " ";
}

.my-linkedin {
  display: inline;
  font-size: 0.8em;
}

#education, #work-experience {
  margin-bottom: 0.5em;
}

.section-title {
  display: inline;
  margin: 0em;
  font-size: 1.2em;
  font-family: 'Alegreya Sans';
}

.name {
  margin-top: 0.5em;
  margin-bottom: 0.2em;
}

.job-title, .degree-title {
  margin-top: 0em;
  margin-bottom: 0em;
  font-style: italic;
}

.employer ul {
  margin: none;
}

.employer .job-description {
  margin-top: 0.5em;
  margin-bottom: 0em;
}

.employer ul {
  margin-top: 0.25em;
  margin-bottom: 0.5em;
}

.employer li {
  list-style: none;
  padding-bottom: 0.1em;
}

.employer li::before {
  content: "— "
}

.employer .relevant-skills {
  margin-top: 0em;
}

#skills ul {
  margin: 0.25em;
  padding-left: 0em;
}

.skills-list {
  list-style-type: none;
  margin: none;
}

.skills-list ul {
  margin-top: 0.25em;
}

.skills-list li {
  display: inline;
}

.skills-list li:not(:last-child)::after {
  content: " · "
}

</style>
