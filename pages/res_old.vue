<template>
  <div class="cv">
    <span id="left"></span>
      <div class="css-1tcp1k1" id="resume">
        <div class="css-qz2fpi">
          <header class="pdf-resume-header css-s1kek">
            <h1 class="css-1u2i2od">Kasyap Munukutla </h1> 
            <h1 class="css-1ltptao">
              <font-awesome-icon :icon="['fab', 'fa-linkedin-in']"  /><a
                href="http://linkedin.com/in/kasyapmunukutla"
                target="_blank"
                rel="noopener noreferrer nofollow ugc"
                class="css-1cuak6k"
                >@linkedin</a>
            </h1> 
          </header>
          <div class="pdf-page-container css-vurnku">
            <div class="pdf-page-column css-vurnku">
              <section class="css-ufg8u0">
                <h1 class="css-1wvg85o">Work Experience</h1>
                <div class="css-19ug6vd">
                  <h2 class="css-1c09ann">
                    <a
                      href="http://samsung.com/"
                      target="_blank"
                      rel="noopener noreferrer nofollow ugc"
                      class="css-1wafmg3"
                      >Samsung, Inc.</a
                    >
                  </h2>
                  <div class="css-sagp40">
                    Sr. Backend Engineer
                    <div class="css-1kvlhqq">| </div>
                    <br class="css-1bmsfsu" /><span
                      >Jul 2022
                      <!-- -->
                      -
                      <!-- -->Current
                    </span>
                  </div>
                  <div class="css-vurnku">
                    <div class="css-vurnku">
                      <p class="css-1swgbfb">
                        <strong class="css-aewju"
                          >Built and deployed microservices for Samsung TV+ streaming service</strong
                        >
                      </p>
                      <ul class="css-1yh09yi">
                        <li class="css-1f0c8ll">
                          Developed and rebuilt several APIs integrating various
                          AWS services into Samsung's channel service to improve
                          code modularity and cleanliness
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
                <div class="css-19ug6vd">
                  <h2 class="css-1c09ann">UChicago CS Department</h2>
                  <div class="css-sagp40">
                    Research Engineer, NLP for Prof. Chenhao Tan
                    <div class="css-1kvlhqq">| </div>
                    <br class="css-1bmsfsu" /><span
                      >Mar 2022
                      <!-- -->
                      -
                      <!-- -->Aug 2022
                    </span>
                  </div>
                  <div class="css-vurnku">
                    <div class="css-vurnku">
                      <p class="css-1swgbfb">
                        <strong class="css-aewju"
                          >Built model predicting whether evoking a given
                          emotion is persuasive using Reddit's
                          <em class="css-mx8wl8">ChangeMyView</em> data</strong
                        >
                      </p>
                      <ul class="css-1yh09yi">
                        <li class="css-1f0c8ll">
                          Data: OP asks for their view to be changed and awards
                          a point to each convincing reply; each thread is a
                          tree with children being responses to parent posts
                        </li>
                        <li class="css-1f0c8ll">
                          Pre-processing: posts are tagged with the
                          corresponding label: convincing or not then tokenized
                          using nltk's tokenizer
                        </li>
                        <li class="css-1f0c8ll">
                          Structure: posts are redacted to reflect only
                          vocabulary for a specific emotion, e.g. fear; word2vec
                          embedding-based negative sampling technique used since
                          99.5% of data is negatively labeled
                        </li>
                        <li class="css-1f0c8ll">
                          Model: BERT-Transformer model implemented using
                          PyTorch, PyTorch Lightning and HuggingFace Transformer
                        </li>
                        <li class="css-1f0c8ll">
                          Results: fear, belligerence, and qualification are all
                          highly predictive (between 75% to 65% compared to
                          baseline of 60%); bluster and confidence are not
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
                <div class="css-19ug6vd">
                  <h2 class="css-1c09ann">UChicago Booth School of Business</h2>
                  <div class="css-sagp40">
                    Research Engineer, Finance for Prof. Jessica Jeffers
                    <div class="css-1kvlhqq">| </div>
                    <br class="css-1bmsfsu" /><span
                      >Jan 2020
                      <!-- -->
                      -
                      <!-- -->Jul 2021
                    </span>
                  </div>
                  <div class="css-vurnku">
                    <div class="css-vurnku">
                      <p class="css-1swgbfb">
                        <strong class="css-aewju"
                          >Conducted research on antitrust policy and economic
                          growth using congressional transcripts, antitrust case
                          records, and business locations</strong
                        >
                      </p>
                      <ul class="css-1yh09yi">
                        <li class="css-1f0c8ll">
                          Produced 5 datasets linking Senate speeches and
                          speakers to campaign contributions and
                          businesses within Senators’ constituencies; instituted
                          processes reducing data ingestion time from months
                          to days making datasets useful for years to come
                        </li>
                        <li class="css-1f0c8ll">
                          built a datastore and ETL pipeline replacing data
                          storage on drives with a well-organized data
                          repository including data dictionaries and entity
                          diagrams; allowed ramp-up of new researchers and
                          increased research speed by 5x
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
                <div class="css-19ug6vd">
                  <h2 class="css-1c09ann">
                    <a
                      href="http://quinstreet.com/"
                      target="_blank"
                      rel="noopener noreferrer nofollow ugc"
                      class="css-1wafmg3"
                      >Quinstreet, Inc.</a
                    >
                  </h2>
                  <div class="css-sagp40">
                    Sr. Manager, Analytics
                    <div class="css-1kvlhqq">| </div>
                    <br class="css-1bmsfsu" /><span
                      >Oct 2019
                      <!-- -->
                      -
                      <!-- -->Aug 2020
                    </span>
                  </div>
                  <div class="css-vurnku">
                    <div class="css-vurnku">
                      <p class="css-1swgbfb">
                        <strong class="css-aewju"
                          >Managed a team of four with oversight of financial
                          and operational reporting as well as being the key
                          strategic input for the email marketing team</strong
                        >
                      </p>
                      <ul class="css-1yh09yi">
                        <li class="css-1f0c8ll">
                          implemented logistic regression and decision tree
                          models to score and filter incoming third-party data
                          sources to better target marketing recipients yielding
                          40% increase in revenues
                        </li>
                        <li class="css-1f0c8ll">
                          automated our processes with Python and HiveQL which
                          resulted in reduced time reporting and manual tasks by
                          60%
                        </li>
                        <li class="css-1f0c8ll">
                          transformed a team reporting in Excel into a team
                          working on prediction and automation using SQL,
                          Hadoop, Tableau, and Python thereby allowing
                          decision-makers daily data refreshes rather than
                          monthly
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
                <div class="css-19ug6vd">
                  <h2 class="css-1c09ann">Stanford University GSB</h2>
                  <div class="css-sagp40">
                    Research Engineer, Computational Sociology for Prof. Amir Goldberg
                    <div class="css-1kvlhqq">| </div>
                    <br class="css-1bmsfsu" /><span
                      >Jul 2016
                      <!-- -->
                      -
                      <!-- -->Mar 2018
                    </span>
                  </div>
                  <div class="css-vurnku">
                    <div class="css-vurnku">
                      <p class="css-1swgbfb">
                        <strong class="css-aewju"
                          >Conducted sociological research on enculturation of
                          employees within organizations using 10M+ records of
                          email data</strong
                        >
                      </p>
                      <ul class="css-1yh09yi">
                        <li class="css-1f0c8ll">
                          produced structured data from variously encoded emails
                          (Gmail, Hotmail, etc.), redacted sensitive information
                          and tokenized data to produce 5 different compact and
                          useable datasets
                        </li>
                        <li class="css-1f0c8ll">
                          built LIWC-based bag-of-words model pipeline to
                          measure linguistic similarity between individuals and
                          groups producing data used by 4+ researcher sub-teams
                        </li>
                        <li class="css-1f0c8ll">
                          created modular and performant data-processing
                          pipeline; improved ETL times from several months to
                          weeks thereby convincing 3+ partners to share
                          sensitive data with the research team
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
                <div class="css-19ug6vd">
                  <h2 class="css-1c09ann">
                    <a
                      href="http://sojern.com/"
                      target="_blank"
                      rel="noopener noreferrer nofollow ugc"
                      class="css-1wafmg3"
                      >Sojern, Inc.
                    </a>
                  </h2>
                  <div class="css-sagp40">
                    Sr. Data Analyst
                    <div class="css-1kvlhqq">|</div>
                    <br class="css-1bmsfsu" /><span
                      >Jan 2016
                      <!-- -->
                      -
                      <!-- -->Oct 2019
                    </span>
                  </div>
                  <div class="css-vurnku">
                    <div class="css-vurnku">
                      <p class="css-1swgbfb">
                        <strong class="css-aewju"
                          >Managed junior colleagues on various analytical and
                          large-scale automation projects including:</strong
                        >
                      </p>
                      <ul class="css-1yh09yi">
                        <li class="css-1f0c8ll">
                          parsing and loading 1K+ client performance reports
                          going from weekly to hourly data refreshes to enable
                          the development of a flagship ML-based campaign
                          optimization product
                        </li>
                        <li class="css-1f0c8ll">
                          managed analytical projects for CEO and exec-team
                          thereby contributing to a successful $100M+
                          fundraising round
                        </li>
                        <li class="css-1f0c8ll">
                          led preparation for Board of Directors meetings on
                          behalf of CEO-staff: maintained official metrics,
                          explained KPI trends, and performed ad-hoc data
                          analysis
                        </li>
                        <li class="css-1f0c8ll">
                          predicted recurrent revenue and customer attrition
                          using regression and decision trees in a stacked model
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
                <div class="css-19ug6vd">
                  <h2 class="css-1c09ann">
                    <a
                      href="http://cardinalhealth.com/"
                      target="_blank"
                      rel="noopener noreferrer nofollow ugc"
                      class="css-1wafmg3"
                      >Cardinal Health</a
                    >
                  </h2>
                  <div class="css-sagp40">
                    Corporate Strategy, Operations Finance, FP&amp;A 
                    <div class="css-1kvlhqq">| </div>
                    <br class="css-1bmsfsu" /><span
                      >Jul 2012
                      <!-- -->
                      -
                      <!-- -->Jan 2016
                    </span>
                  </div>
                  <div class="css-vurnku">
                    <div class="css-vurnku">
                      <p class="css-1swgbfb">
                        Consultant, Corporate Strategy | Jun. 2014 - Jan 2016
                      </p>
                      <p class="css-1swgbfb">
                        Sr. Analyst, Finance | Jul. 2012 - Jun. 2014
                      </p>
                    </div>
                  </div>
                </div>
              </section>
              <section class="css-ufg8u0">
                <h1 class="css-1wvg85o">Education</h1>
                <div class="css-19ug6vd">
                  <h2 class="css-1c09ann">
                    <a
                      href="http://uchicago.edu/"
                      target="_blank"
                      rel="noopener noreferrer nofollow ugc"
                      class="css-1wafmg3"
                      >University of Chicago</a
                    >
                  </h2>
                  <div class="css-sagp40">
                    MS, Computer Science with specialization in Machine Learning
                    <div class="css-1kvlhqq">| </div>
                    <br class="css-1bmsfsu" /><span>Jun 2022</span>
                  </div>

                </div>
                <div class="css-19ug6vd">
                  <h2 class="css-1c09ann">University of Virginia</h2>
                  <div class="css-sagp40">
                    BA, Economics
                    <div class="css-1kvlhqq">| </div>
                    <br class="css-1bmsfsu" /><span
                      > May 2012
                    </span>
                  </div>
                </div>
              </section>
              <section class="css-gz6vba">
                <h1 class="css-1wvg85o">Skills</h1>
                <ul class="css-10c7d0b">
                  <li class="css-1w3ob58">Python (Numpy/PyTorch/Pandas/Flask/FastAPI)</li>
                  <li class="css-1w3ob58">SQL</li>
                  <li class="css-1w3ob58">C++</li>
                  <li class="css-1w3ob58">C</li>
                  <li class="css-1w3ob58">Rust</li>
                  <li class="css-1w3ob58">Go</li>
                  <li class="css-1w3ob58">Haskell</li>
                  <li class="css-1w3ob58">Docker/Docker Compose</li>
                  <li class="css-1w3ob58">Latex</li>
                  <li class="css-1w3ob58">Web Dev (JS/HTML/CSS/VueJS/Nuxt)</li>
                  <li class="css-1w3ob58">BigQuery</li>
                  <li class="css-1w3ob58-end">AWS</li>
                </ul>
              </section>
            </div>
          </div>
        </div>
      </div>
    <span id="right">
    </span>
  </div>
</template>

<script>
definePageMeta({
  layout: "default",
});

// Put the tag below in the right span
//<!--<button @click="exportToPDF" class="pdfButton"><span>pdf</span></button>-->
export default {
  name: "Resume",
  methods: {
     exportToPDF() {
        console.log("Running pdfer.");
        html2pdf(document.getElementById("resume"), {
        filename: "resume.pdf",
      });
      console.log("Running pdfer again.");
    },
  },
};
</script>


<style scoped>
div {
  font-family: 'Alegreya Sans', sans-serif;
}
</style>
<style data-emotion-css="be9d8r">

.css-be9d8r {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  background-color: #e4ebf1;
  min-height: 100vh;
  min-width: 100vw;
}
</style>

<style data-emotion-css="1jmx1h9">
.css-1jmx1h9 {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  padding-left: 0;
  padding-right: 0;
  margin-left: auto;
  margin-right: auto;
  max-width: 1084px;
  padding-bottom: 0;
}
@media screen and (min-width: 768px) {
  .css-1jmx1h9 {
    padding-left: 0;
    padding-right: 0;
    padding-bottom: 0;
  }
}
@media screen and (min-width: 1024px) {
  .css-1jmx1h9 {
    padding-left: 42px;
    padding-right: 42px;
    padding-bottom: 42px;
  }
}
</style>
  <style data-emotion-css="1qbz534">
.css-1qbz534 {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  display: -webkit-box;
  display: -webkit-flex;
  display: -ms-flexbox;
  display: flex;
  -webkit-box-pack: justify;
  -webkit-justify-content: space-between;
  -ms-flex-pack: justify;
  justify-content: space-between;
  -webkit-align-items: center;
  -webkit-box-align: center;
  -ms-flex-align: center;
  align-items: center;
  padding-left: 16px;
  padding-right: 16px;
  min-height: 0;
}
@media screen and (min-width: 768px) {
  .css-1qbz534 {
    padding-left: 16px;
    padding-right: 16px;
    min-height: 0;
  }
}
@media screen and (min-width: 1024px) {
  .css-1qbz534 {
    padding-left: 0;
    padding-right: 0;
    min-height: 42px;
  }
}
</style>
  <style data-emotion-css="1bnx3o6">
.css-1bnx3o6 {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  padding-top: 14px;
  padding-bottom: 14px;
  color: rgba(0, 0, 0, 0.5);
}
</style>
  <style data-emotion-css="1lummbz">
.css-1lummbz {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  -webkit-text-decoration: none;
  text-decoration: none;
  font-size: 15px;
  font-weight: 700;
  color: rgba(0, 0, 0, 0.5);
}
@media (hover: hover) {
  .css-1lummbz:hover {
    -webkit-text-decoration: none;
    text-decoration: none;
    -webkit-transform: none;
    -ms-transform: none;
    transform: none;
  }
}
</style>
  <style data-emotion-css="1yqdrc0">
.css-1yqdrc0 {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  -webkit-text-decoration: none;
  text-decoration: none;
  font-size: 15px;
  font-weight: 400;
  color: rgba(0, 0, 0, 0.5);
}
@media (hover: hover) {
  .css-1yqdrc0:hover {
    -webkit-text-decoration: none;
    text-decoration: none;
    -webkit-transform: none;
    -ms-transform: none;
    transform: none;
  }
}
</style>
  <style data-emotion-css="10nf7hq">
.css-10nf7hq {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  display: -webkit-box;
  display: -webkit-flex;
  display: -ms-flexbox;
  display: flex;
  -webkit-align-items: center;
  -webkit-box-align: center;
  -ms-flex-align: center;
  align-items: center;
}
</style>
  <style data-emotion-css="rcrn2l">
.css-rcrn2l {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  stroke: rgba(0, 0, 0, 0.5);
  fill: none;
  margin-left: 8px;
}
.css-rcrn2l path {
  stroke-width: 1;
}
</style>
<style data-emotion-css="1tcp1k1">
.css-1tcp1k1 {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  margin: 0 auto;
  width: 100%;
  max-width: 1000px;
  background-color: #fff;
  overflow: hidden;
  min-height: auto;
}
</style>
  <style data-emotion-css="qz2fpi">
.css-qz2fpi {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  margin: 0 auto;
  width: 100%;
  max-width: 800px;
  padding-left: 24px;
  padding-right: 24px;
  padding-top: 24px;
  padding-bottom: 24px;
}
@media screen and (min-width: 768px) {
  .css-qz2fpi {
    padding-left: 60px;
    padding-right: 60px;
    padding-top: 20px;
    padding-bottom: 60px;
  }
}
</style>
  <style data-emotion-css="s1kek">
.css-s1kek {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  margin-bottom: 12px;
  position: relative;
}
</style>
  <style data-emotion-css="1ltptao">
.css-1ltptao {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  font-family: "Alegreya Sans", sans-serif;
  line-height: 1.4;
  font-weight: 400;
  color: #000000;
  font-size: 13px;
  margin-bottom: 6px;
}
</style>
  <style data-emotion-css="1cuak6k">
.css-1cuak6k {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  color: gray;
  -webkit-text-decoration: inherit;
  text-decoration: inherit;
  font-size: inherit;
  font-weight: inherit;
  overflow-wrap: break-word;
}
.css-1cuak6k:hover {
  opacity: 0.7;
}
</style>
  <style data-emotion-css="88lgb3">
.css-88lgb3 {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  font-family: "Alegreya Sans", sans-serif;
  line-height: 1.4;
  font-weight: 400;
  color: #000000;
  font-size: 14px;
  margin-bottom: 6px;
  color: #676868;
}
</style>
  <style data-emotion-css="vurnku">
.css-vurnku {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
}
</style>
  <style data-emotion-css="ufg8u0">
.css-ufg8u0 {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  margin-bottom: 8px;
}
</style>
<style data-emotion-css="1u2i2od">
.css-1u2i2od {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  font-family: "Alegreya Sans SC Bold", sans-serif;
  font-weight: 700;
  color: #000000;
  line-height: 1.2;
  text-transform: uppercase;
  font-size: 18px;
  -webkit-letter-spacing: 1px;
  -moz-letter-spacing: 1px;
  -ms-letter-spacing: 1px;
  letter-spacing: 0px;
  margin-bottom: 16px;
  display: inline;
}
</style>
  <style data-emotion-css="1wvg85o">
.css-1wvg85o {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  font-family: "Alegreya Sans SC Bold", sans-serif;
  font-weight: 700;
  color: #000000;
  line-height: 1.4;
  text-transform: uppercase;
  font-size: 12px;
  -webkit-letter-spacing: 1.2px;
  -moz-letter-spacing: 1.2px;
  -ms-letter-spacing: 1.2px;
  letter-spacing: 1.2px;
  margin-bottom: 16px;
}
</style>
  <style data-emotion-css="19ug6vd">
.css-19ug6vd {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  margin-bottom: 12px;
  page-break-inside: avoid;
}
</style>
  <style data-emotion-css="1c09ann">
.css-1c09ann {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  font-family: "Alegreya Sans Medium", sans-serif;
  font-weight: 900;
  line-height: 1;
  color: #000000;
  line-height: 20px;
  text-transform: uppercase;
  font-size: 16px;
  -webkit-letter-spacing: 0.07em;
  -moz-letter-spacing: 0.07em;
  -ms-letter-spacing: 0.07em;
  letter-spacing: 0.07em;
  margin-bottom: 4px;
  display: inline;
}
</style>
  <style data-emotion-css="1wafmg3">
.css-1wafmg3 {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  color: inherit;
  -webkit-text-decoration: inherit;
  text-decoration: inherit;
  font-size: inherit;
  font-weight: inherit;
}
.css-1wafmg3:hover {
  opacity: 0.7;
}
</style>
  <style data-emotion-css="sagp40">
.css-sagp40 {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  color: #3e3e3e;
  font-size: 16px;
  line-height: 20px;
  margin-bottom: 4px;
  font-family: "Alegreya Sans SC", sans-serif;
  font-style: italic;
}
</style>
  <style data-emotion-css="1kvlhqq">
.css-1kvlhqq {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  display: none;
}
@media (min-width: 500px) {
  .css-1kvlhqq {
    display: inline;
  }
}
</style>
  <style data-emotion-css="1bmsfsu">
.css-1bmsfsu {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  display: inline;
}
@media (min-width: 500px) {
  .css-1bmsfsu {
    display: none;
  }
}
</style>
  <style data-emotion-css="1swgbfb">
.css-1swgbfb {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  font-family: "Alegreya Sans", sans-serif;
  line-height: 1.5;
  font-weight: 400;
  color: #000000;
  font-size: 12px;
  margin-bottom: 4px;
}
</style>
<style data-emotion-css="aewju">
.css-aewju {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  font-family: 'Alegreya Sans';
  font-size: 1.1em;
  font-style: bold;
}
</style>
  <style data-emotion-css="1yh09yi">
.css-1yh09yi {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  padding: 0;
}
</style>
  <style data-emotion-css="1f0c8ll">
.css-1f0c8ll {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  font-family: "Alegreya Sans", sans-serif;
  line-height: 1.5;
  font-weight: 400;
  color: #000000;
  font-size: 12px;
  margin-bottom: 4px;
  margin-left: 30px;
  padding-left: 0;
}
</style>
  <style data-emotion-css="mx8wl8">
.css-mx8wl8 {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  font-style: italic;
}
</style>
  <style data-emotion-css="gz6vba">
.css-gz6vba {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  margin-bottom: 40px;
  page-break-inside: avoid;
}
</style>
  <style data-emotion-css="10c7d0b">
.css-10c7d0b {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  list-style: none;
  margin: 0;
  padding: 0;
}

</style>
<style data-emotion-css="1w3ob58">
.css-1w3ob58 {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  font-family: "Alegreya Sans", sans-serif;
  line-height: 0.1;
  font-weight: 400;
  color: #000000;
  margin-bottom: 0px;
  display: inline-block;
  color: #000000;
  padding-top: 0px;
  padding-bottom: 0px;
  margin-right: 2px;
  margin-left: 2px;
  margin-bottom: 8px;
  font-size: 9px;
  text-decoration: none;
}

.css-1w3ob58-end {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  font-family: "Alegreya Sans", sans-serif;
  line-height: 0.1;
  font-weight: 400;
  color: #000000;
  font-size: 9px;
  display: inline-block;
  color: #000000;
  padding-top: 5px;
  padding-bottom: 0px;
  margin-right: 12px;
  margin-bottom: 12px;
  font-size: 12px;
  text-decoration: none;
}

.css-1w3ob58::after{
  content: "\00B7";
  font-size: 20px;
  margin-right: 2px;
  margin-left: 2px;
}
</style>
  <style data-emotion-css="1si0cz4">
.css-1si0cz4 {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  display: -webkit-box;
  display: -webkit-flex;
  display: -ms-flexbox;
  display: flex;
  -webkit-align-items: center;
  -webkit-box-align: center;
  -ms-flex-align: center;
  align-items: center;
  -webkit-box-pack: center;
  -webkit-justify-content: center;
  -ms-flex-pack: center;
  justify-content: center;
  margin-top: 48px;
  padding-bottom: 24px;
}
@media screen and (min-width: 768px) {
  .css-1si0cz4 {
    margin-top: 120px;
    padding-bottom: 0;
  }
}
</style>
  <style data-emotion-css="1cjs9fo">
.css-1cjs9fo {
  box-sizing: border-box;
  margin: 0;
  min-width: 0;
  display: -webkit-inline-box;
  display: -webkit-inline-flex;
  display: -ms-inline-flexbox;
  display: inline-flex;
  color: rgba(42, 63, 251, 1);
  -webkit-text-decoration: none;
  text-decoration: none;
  -webkit-transition: all 0.2s linear;
  transition: all 0.2s linear;
  font-size: 12px;
  font-weight: 400;
  color: rgba(0, 0, 0, 0.5);
  padding-top: 6px;
  padding-bottom: 6px;
  padding-left: 8px;
  padding-right: 8px;
}
@media (hover: hover) {
  .css-1cjs9fo:hover {
    -webkit-text-decoration: none;
    text-decoration: none;
    -webkit-transform: scale(1.05);
    -ms-transform: scale(1.05);
    transform: scale(1.05);
    -webkit-filter: brightness(140%);
    filter: brightness(140%);
  }
  .css-1cjs9fo:active {
    -webkit-transform: scale(1);
    -ms-transform: scale(1);
    transform: scale(1);
  }
}
@media screen and (min-width: 768px) {
  .css-1cjs9fo {
    font-size: 13px;
    padding-left: 12px;
    padding-right: 12px;
  }
}
</style>
  <style type="text/css">
iframe#_hjRemoteVarsFrame {
  display: none !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
  pointer-events: none !important;
}
</style>
  <style scoped>
.cv {
  width: 100%;
  text-align: left;
  display: grid;
  grid-template-columns: 5% 90% 5%;
}
h1, h2, h3, section {
    display: inline;
}

#right{
  padding-top: 20px;
}

.button {
  display: inline-block;
  border-radius: 4px;
  border: none;
  text-align: center;
  font-size: 0.75em;
  transition: all 0.5s;
  cursor: pointer;
  margin: 5px;
  padding-right: 10px;
  padding-left: 10px;
  background-color: transparent;
  color: blue;
  font-family: "Alegreya Sans SC", sans-serif;
}

.button span {
  cursor: pointer;
  display: inline-block;
  position: relative;
  transition: 0.5s;
}

.button span:after {
  content: "\00bb";
  position: absolute;
  opacity: 0;
  top: 0;
  right: -20px;
  transition: 0.5s;
}

.button:hover span {
  padding-right: 20px;
}

.button:hover span:after {
  opacity: 1;
  right: 0;
}
</style>
