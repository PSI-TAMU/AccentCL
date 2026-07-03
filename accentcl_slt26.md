AccentCL: Robust Accent Classification with
Incremental Expansion
Anonymous Authors
Paper under double-blind review
Abstract—Accent classifiers are typically trained with a fixed variation can be substantial — differences between Southern
labelinventoryandcannotaccommodatenewaccentcategoriesas andNortheasternUSaccents,forinstance,areoftenmorepro-
new data becomes available. Moreover, accented speech corpora
nounced than those between American and Canadian English.
often exhibit substantial class imbalance and/or domain shift
Second, accent label distributions tend to be long-tailed, with
due to differences in recording conditions across corpora. We
present AccentCL, a class-incremental learning framework for some categories represented by substantially fewer samples
English accent classification that is robust to class imbalance than others. Third, training samples are frequently pooled
and cross-corpus domain shift. AccentCL extracts multi-layer across multiple corpora that differ in recording conditions,
representations from a frozen Whisper-Large-v3 encoder, opti-
speaker demographics, and collection protocols, introducing
mizedwithanimbalance-awarecross-entropylosstoreducebias
corpus-level domain shift. Finally, these models do not allow
towardthemajorityaccentclassesandadomainmeanalignment
loss that minimizes distributional mean shift across training newaccentcategoriestobeaddedafteraclassifierhasalready
corpora. The label space is then expanded via replay-based been deployed. Retraining from scratch to accommodate new
continual learning, using the frozen base model for knowledge classes is computationally costly and often impractical when
retentionandanold-to-newmarginlosstoreduceoverprediction
theoriginaltrainingdataarenolongeravailableorwhenonly
on newly added classes. On a five-class accent classification
limited replay of old-class samples can be retained.
task, AccentCL achieves 77.1% balanced accuracy and a 76.9%
macro-averagedF1score.Wefurtherevaluatethemodel’sability
We propose ACCENTCL, a framework for robust, class-
toincrementallyincorporatetwonewaccentcategories:Spanish- incrementalEnglishaccentclassification.AccentCLfirsttrains
accented and Chinese-accented English. When adding Spanish- anaccentclassifierbymergingcloselyrelatedfine-grainedac-
accented English to the pretrained model, AccentCL attains an centlabelsintobroaderregionalgroupsthatreflectgeographic
F1 of 83.3% on the new class while retaining 77.3% balanced
influences, reducing ambiguity arising from acoustically simi-
accuracyonthebaseclasses.WhensubsequentlyaddingChinese-
larorinconsistentlyannotatedlabels.Theclassifieroperateson
accented English, it achieves 61.8% F1 on the new class while
preserving 77.6% balanced accuracy on the previously learned multi-layer representations extracted from a frozen Whisper-
classes.TheseresultsshowthatAccentCLenablesrobustregional Large-v3 encoder and is trained with class-prior adjustment
accent classification while allowing new accent categories to be andcorpus-meanalignmenttoimproverobustnesstoclassim-
added without full retraining.
balanceandcross-corpusdomainshift,respectively.AccentCL
Index Terms—accent classification, continual learning
then expands the label space by incorporating new accent
I. INTRODUCTION classes, using a small replay memory of previously learned
accentsstratifiedacrosscorpora.Duringcontinualtraining,the
The ability to identify regional or non-native varieties of
updated model is regularized against the frozen base classifier
speech enables a range of applications, including speaker
forknowledgeretentionandusesanold-to-newmarginlossto
population analysis, evaluation of accented speech synthe-
reduceoverpredictionofnewlyaddedclassesonsamplesfrom
sis/conversion systems [1], [2], and measurement of perfor-
previously learned accents. Together, these components allow
mance disparities in downstream speech models [3]. Further,
the classifier to learn new accent categories while preserving
accent-related variation carries acoustic and phonetic cues
performance on previously learned ones.
(e.g.,inpronunciation,rhythm,andprosody)thatcanaffectthe
Our main contributions are:
performance of automatic speech recognition, voice conver-
sion, and spoken language understanding systems. As speech • We formulate English accent classification as a robust
class-incremental learning problem, one that must con-
technologies are increasingly deployed across diverse speaker
tendwithambiguousaccentboundaries,classimbalance,
populations, developing accent classifiers that are both robust
domain heterogeneity, and limited replay.
and extensible has become an important research direction.
Recent accent classifiers [4], [5] benefit from large pre- • We show that multi-layer representations from a frozen
speech foundation model can be adapted for robust
trained speech encoders and public speech corpora, but often
regional accent classification through imbalance-aware
rely on strong assumptions about the underlying data. First,
training and class-conditional domain mean alignment.
accent labels are typically defined along political or national
boundaries (e.g., American vs. Canadian English) rather than • We introduce a replay-based continual training objective
that combines domain-stratified memory, frozen-model
geographic or cultural lines, even though intra-national accent
retention, and an old-to-new margin loss, enabling new
Projectpage:https://anonymized0826.github.io/AccentCL/ accent classes to be added while preserving performance

Phase 1: Base Training
|     | Input Speech |     |     | Whisper-large-v3 encoder |     |     |           |     |     |                |            |     |     |     |
| --- | ------------ | --- | --- | ------------------------ | --- | --- | --------- | --- | --- | -------------- | ---------- | --- | --- | --- |
|     |              |     |     |                          |     |     | 𝐻∈ℝ𝑇×𝐿×𝑑′ |     |     | C l a s s ifie | r  H e a d |     |     |     |
|     |              |     |     |                          |     |     |           |     |     | 𝑊 ∈ ℝ          | 𝑛 𝑒× 𝑑     |     |     |     |
|     |              |     |     | Feature Projection       |     |     |           |     |     | 𝑏 𝑎 𝑠 𝑒        | 𝑏𝑎 𝑠       |     |     |     |
𝑙𝑜𝑔𝑖𝑡𝑠𝑎𝑏𝑎𝑠𝑒
|     |     |     |     | Transformer Block 1 |     |     | Multi-layer feature fusion |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ------------------- | --- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- |
(layer-wise projection + concat)
|     |     |     |     |     | …   |     |     |     | Training Objectives: ℒ𝑏𝑎𝑠𝑒=ℒ𝐶𝐿𝑆+𝛽𝑎𝑙𝑖𝑔𝑛ℒ𝑎𝑙𝑖𝑔𝑛 |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------------------- | --- | --- | --- | --- | --- |
Log-Mel Spectrogram
Attentive S ta t s Pooling +
|     |     |     |     | Transformer Block ℓ𝑖 |     |     | M   | L P |     | Phase 2: Continual Expansion |     |     |     |     |
| --- | --- | --- | --- | -------------------- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | --- | --- |
𝐴𝑐𝑐𝑒𝑛𝑡𝐸𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔∈ℝ𝑑
|     |     |     |     | Transformer Block ℓ𝑗 |     |     |     |     |     | Classifier |     |     |     |     |
| --- | --- | --- | --- | -------------------- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- |
…
Old Classifier Head
|     |     |     |     |     |     |     |     |     |     | 𝑊𝑜𝑙𝑑∈ℝ𝑛𝑜𝑙𝑑×𝑑 |     | l𝑜𝑔𝑖𝑡𝑠𝑎𝑜𝑙𝑑 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ---------- | --- | --- |
Transformer Block 32
New Classifier Head
𝑤𝑛𝑒𝑤∈ℝ1×𝑑
l𝑜𝑔𝑖𝑡𝑎𝑛𝑒𝑤
|     | Accent  | DomainMean Alignment (ℒ𝑎𝑙𝑖𝑔𝑛) |     |     |     | Retention Loss (ℒ𝑟𝑒𝑡) |     |     | Training Objectives:  |     |     |     |     |     |
| --- | ------- | ----------------------------- | --- | --- | --- | --------------------- | --- | --- | --------------------- | --- | --- | --- | --- | --- |
label 𝑦
Source A  Source B  ℒ𝐶𝐿=ℒ𝐶𝐿𝑆+𝛽𝑎𝑙𝑖𝑔𝑛ℒ𝑎𝑙𝑖𝑔𝑛+𝛽𝑟𝑒𝑡ℒ𝑟𝑒𝑡+𝛽𝑜𝑙𝑑−𝑛𝑒𝑤ℒ𝑜𝑙𝑑−𝑛𝑒𝑤
mean emb𝜇𝑦,𝐴 mean emb𝜇𝑦,𝐵
Source
|     | la b el  𝑑 ∈  |     |                  |     |                | 𝑎𝑏     |                    | 𝑎𝑜   |                                   |     |                                      |                    |     |     |
| --- | ------------- | --- | ---------------- | --- | -------------- | ------ | ------------------ | ---- | --------------------------------- | --- | ------------------------------------ | ------------------ | --- | --- |
|     | { 𝐴 ,𝐵 , 𝐶… } |     |                  |     | 𝑝𝑏𝑎𝑠𝑒=𝑠𝑜𝑓𝑡𝑚𝑎𝑥( |        | 𝑎𝑠𝑒) 𝑝𝑜𝑙𝑑=𝑠𝑜𝑓𝑡𝑚𝑎𝑥( | 𝑙𝑑)  |                                   |     |                                      |                    |     |     |
|     |               |     |                  |     |                |        | 𝜏                  | 𝜏    | Old-to-new Margin Loss (ℒ𝑜𝑙𝑑−𝑛𝑒𝑤) |     |                                      |                    |     |     |
|     |               |     |                  |     |                | 𝜏2     |                    |      |                                   |     | Assume an old sample with accent y,  |                    |     |     |
|     |               |     | class mean emb𝜇𝑦 |     | ℒ𝑟𝑒𝑡=          | 𝐵𝑚𝑒𝑚𝑖∈ | ෍ D𝐾𝐿(𝑝𝑖 𝑏𝑎𝑠𝑒||𝑝𝑖  | 𝑜𝑙𝑑) |                                   |     | we want:                             |                    |     |     |
|     |               |     |                  |     |                |        | 𝐵𝑚𝑒𝑚               |      |                                   |     |                                      | 𝑎𝑜𝑙𝑑,𝑦−𝑎𝑛𝑒𝑤>𝑚𝑎𝑟𝑔𝑖𝑛 |     |     |
Fig. 1: Model architecture of AccentCL. A frozen Whisper-Large-v3 encoder produces multi-layer features that are fused into
an accent embedding. Phase 1 trains the regional classifier with domain mean alignment, and Phase 2 expands it with replay,
| retention, | and an old-to-new |     | margin | loss. |     |     |     |     |     |     |     |     |     |     |
| ---------- | ----------------- | --- | ------ | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
on old ones. wisestructureofspeechfoundationmodels:priorworkshows
|     |     |             |     |     |     |     | that    | different | encoder | layers    | capture   | different | types | of infor-  |
| --- | --- | ----------- | --- | --- | --- | --- | ------- | --------- | ------- | --------- | --------- | --------- | ----- | ---------- |
|     | II. | RELATEDWORK |     |     |     |     |         |           |         |           |           |           |       |            |
|     |     |             |     |     |     |     | mation, | including |         | acoustic, | phonetic, | speaker,  | and   | linguistic |
A. Accent classification with speech foundation models properties[8].Sinceaccentcuesspanpronunciation,phoneme
Accent classification aims to identify a speaker’s regional realization, rhythm, and prosody, AccentCL aggregates repre-
or non-native accent from speech. Recent systems com- sentations across multiple Whisper encoder layers rather than
|         |                     |      |        |                  |           |      | relying | on  | the final | layer | alone. |     |     |     |
| ------- | ------------------- | ---- | ------ | ---------------- | --------- | ---- | ------- | --- | --------- | ----- | ------ | --- | --- | --- |
| monly   | build on pretrained |      | speech | representations, | including |      |         |     |           |       |        |     |     |     |
| wav2vec | 2.0 [6], XLS-R      | [7], | WavLM  | [8], and         | Whisper   | [9]. |         |     |           |       |        |     |     |     |
These models yield contextual frame-level representations B. Class-Incremental Learning
| that capture | acoustic, | phonetic, | and | speaker-related | informa- |     |     |     |     |     |     |     |     |     |
| ------------ | --------- | --------- | --- | --------------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tion [10], which can then be aggregated via utterance-level Class-incrementallearningaddressestheproblemofextend-
ingamodelwithnewclasseswhilepreservingperformanceon
| pooling | and passed to | a classification |     | head. |     |     |     |     |     |     |     |     |     |     |
| ------- | ------------- | ---------------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
CommonAccent [4] is a pretrained English accent clas- previouslylearnedones.Unliketask-incrementallearning[12],
sifier trained on CommonVoice [11] with a fixed 16-class the model must predict over a unified label space at test
time,whichmakesthissettingespeciallypronetocatastrophic
| accent | inventory, but | its single-domain |     | training | data and | fine- |     |     |     |     |     |     |     |     |
| ------ | -------------- | ----------------- | --- | -------- | -------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
grained label space may limit how well it generalizes to forgetting and bias toward recently added classes.
Existingmethodstypicallyreduceforgettingthroughreplay,
| other domains. | In our | evaluation, |     | many low-resource |     | and |     |     |     |     |     |     |     |     |
| -------------- | ------ | ----------- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
out-of-domain accents are confused with related labels or whichmixesstoredold-classexampleswithnew-classdata,or
absorbed into dominant classes, which motivates our use of a through regularization and distillation, which encourage the
regional label mapping together with domain-aware training. updated model to stay close to the previous model [13]–[16].
Voxlect [5] introduces a larger mixed-corpus benchmark for This line of work is directly relevant to accent classification,
|     |     |     |     |     |     |     | since | new | accent | categories | may | need to | be introduced | after |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | ------ | ---------- | --- | ------- | ------------- | ----- |
Englishdialectandregionallanguageclassification,evaluating
several speech foundation models with LoRA fine-tuning; its a classifier has already been trained. However, fine-tuning
strongest configuration, based on Whisper-Large-v3, serves directly on the new accent can bias the classifier toward that
as a strong reference model for accent-related classification. class and degrade previously learned accents.
However, Voxlect remains a fixed-label classifier and is not AccentCL supports incremental accent-class expansion by
built to accommodate new accent categories after training. combiningnew-accentdatawithamemoryofpreviousaccents
AccentCLaddressesthislimitationbyfirsttrainingarobust drawn from multiple corpora. During training, it preserves
regional accent classifier and then extending it through class- the earlier model’s behavior on old-accent examples and
incremental learning. Our model also draws on the layer- suppresses the tendency to misclassify old accents as the

newly added class, limiting degradation on previously learned pooled representation is fed into an MLP to obtain a compact
accents. accent embedding, which is then passed to a linear classifier
|     |       |                         |     |      |        |                   |     | to predict  | the regional              |            | accent | label. |                   |          |         |
| --- | ----- | ----------------------- | --- | ---- | ------ | ----------------- | --- | ----------- | ------------------------- | ---------- | ------ | ------ | ----------------- | -------- | ------- |
|     |       | III. PROBLEMFORMULATION |     |      |        |                   |     |             |                           |            |        |        |                   |          |         |
|     |       |                         |     |      |        |                   |     | b) Training |                           | objective: |        | For    | each              | training | example |
| We  | study | accent classification   |     | as a | robust | class-incremental |     |             |                           |            |        |        |                   |          |         |
|     |       |                         |     |      |        |                   |     | (x ,y ,d    | ),themodelproduceslogitsa |            |        |        | andanaccentembed- |          |         |
learning problem. Given an accent classifier trained on a set i i i i
|     |     |     |     |     |     |     |     | ding z . | To account | for | class | imbalance, | we  | use logit-adjusted |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | --- | ----- | ---------- | --- | ------------------ | --- |
i
of “old” accent classes Y old , the goal is to add a “new” R|Ybase|
|        |       |         |            |     |             |     |         | cross-entropy | [18]. | Let | π ∈ |     | be the | empirical | class- |
| ------ | ----- | ------- | ---------- | --- | ----------- | --- | ------- | ------------- | ----- | --- | --- | --- | ------ | --------- | ------ |
| accent | class | y while | preserving |     | performance | on  | the old |               |       |     |     |     |        |           |        |
new prior vector in the training set. The classification loss is:
| classes. | Before | each | incremental | step, | we  | construct | a replay |     |     |     |          |     |     |     |     |
| -------- | ------ | ---- | ----------- | ----- | --- | --------- | -------- | --- | --- | --- | -------- | --- | --- | --- | --- |
|          |        |      |             |       |     |           |          |     |     | 1   | (cid:88) |     |     |     |     |
memory D from the previously learned classes. During L = ℓ (a +τlogπ,y ),
|     |     | mem |     |     |     |     |     |     | cls |     | CE  | i   |     | i   | (3) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
incremental training, the updated classifier f is trained on B
|                                 |        |           |             |                |                      | θ              |          |               |            |                | i∈B            |        |        |             |          |
| ------------------------------- | ------ | --------- | ----------- | -------------- | -------------------- | -------------- | -------- | ------------- | ---------- | -------------- | -------------- | ------ | ------ | ----------- | -------- |
| D train                         | = D    | new ∪ D   | mem , where | D              | new contains         | training       | ut-      |               |            |                |                |        |        |             |          |
|                                 |        |           |             |                |                      |                |          | where τ       | controls   | the strength   |                | of the | logit  | adjustment. | Setting  |
| terances                        | from   | the newly | added       | accent         | class.               | Each           | training |               |            |                |                |        |        |             |          |
|                                 |        |           |             |                |                      |                |          | τ =0 recovers |            | standard       | cross-entropy. |        |        |             |          |
| exampleisrepresentedas(x        |        |           |             | ,y ,d ),wherex |                      | isanutterance, |          |               |            |                |                |        |        |             |          |
|                                 |        |           |             | i i i          |                      | i              |          |               |            |                |                |        |        |             |          |
|                                 |        |           |             |                |                      |                |          | We also       | regularize |                | the learned    |        | accent | embedding   | space    |
| y ∈Y                            | is the | accent    | label, and  | d is           | the domain/corpus    |                | label.   |               |            |                |                |        |        |             |          |
| i                               |        |           |             | i              |                      |                |          | with a domain |            | mean alignment |                | (DMA)  | loss.  | Since       | the same |
| Theobjectiveistolearnafunctionf |        |           |             |                | thatrecognizesthenew |                |          |               |            |                |                |        |        |             |          |
θ
|        |        |               |             |        |              |     |         | regional       | accent      | may appear | in         | multiple     | datasets  | with        | different  |
| ------ | ------ | ------------- | ----------- | ------ | ------------ | --- | ------- | -------------- | ----------- | ---------- | ---------- | ------------ | --------- | ----------- | ---------- |
| accent | class, | retains       | performance | on     | old classes, | and | remains |                |             |            |            |              |           |             |            |
|        |        |               |             |        |              |     |         | recording      | conditions, | speaker    |            | pools,       | or corpus | collection  | pro-       |
| robust | across | heterogeneous |             | speech | corpora.     |     |         |                |             |            |            |              |           |             |            |
|        |        |               |             |        |              |     |         | tocols, this   | term        | encourages |            | embeddings   | to        | remain      | consistent |
|        |        |               | IV.         | METHOD |              |     |         |                |             |            |            |              |           |             |            |
|        |        |               |             |        |              |     |         | across sources |             | while      | preserving | accent-class |           | separation. | For        |
A. Phase 1: Base regional accent classifier accent class y and domain d, let µ denote the mean
y,d
The base classifier provides recognition capabilities on the embeddingofsamples,andletµ y denotethemeanembedding
|        |         |         |           |            |          |                 |           | of all samples   | with | label | y in   | the mini-batch. |     | The     | alignment |
| ------ | ------- | ------- | --------- | ---------- | -------- | --------------- | --------- | ---------------- | ---- | ----- | ------ | --------------- | --- | ------- | --------- |
| “old”  | accents | before  | expanding | to new     | classes. | We              | define an |                  |      |       |        |                 |     |         |           |
|        |         |         |           |            |          |                 |           | loss is computed |      | over  | active | class-domain    |     | groups: |           |
| accent | label   | space Y | by        | collapsing | the      | 16 fine-grained |           |                  |      |       |        |                 |     |         |           |
base
accent categories in CommonAccent [4] into 5 broad accent 1 (cid:88) ∥2,
|     |     |     |     |     |     |     |     |     | L   | =   |     | ∥µ  | −µ  |     | (4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
categories:NorthAmerican(AmericanandCanadian),British align |G| y,d y 2
Isles (British, Irish, Scottish, and Welsh), Australasian (Aus- (y,d)∈G
tralianandNewZealand),SouthAsian(Indian),andSoutheast where G is the set of active class-domain groups in the mini-
Asian (Malaysian and Singaporean). This grouping preserves batch,definedasgroupswithatleastmsampleswhoseaccent
broad regional structure while avoiding unreliable distinctions m=2
|     |     |     |     |     |     |     |     | class is represented |     | in  | at least | two domains. |     | We set | in  |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------- | --- | --- | -------- | ------------ | --- | ------ | --- |
between fine-grained labels with limited training data. our experiments. The final base-training objective is:
|     | a) Multi-layeraccentencoder: |     |     |     | Givenaninputwaveform |     |     |     |     |     |     |     |     |     |     |
| --- | ---------------------------- | --- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |                              |     |     |     |                      |     |     |     |     | L   | =L  | +β  | L   | ,   | (5) |
x, we use the encoder of a pretrained Whisper-Large-v3 base cls align align
| model      | [9] as | the acoustic | feature  | extractor. |                | We train | only the |               |          |     |              |     |               |     |       |
| ---------- | ------ | ------------ | -------- | ---------- | -------------- | -------- | -------- | ------------- | -------- | --- | ------------ | --- | ------------- | --- | ----- |
|            |        |              |          |            |                |          |          | where β align | controls |     | the strength | of  | the alignment |     | term. |
| downstream |        | projection,  | pooling, | and        | classification |          | modules. |               |          |     |              |     |               |     |       |
Motivated by prior findings that intermediate layers of speech B. Phase 2: Continual expansion
| foundation |     | models retain | useful | acoustic | and | phonetic | infor- |        |       |        |        |           |     |            |         |
| ---------- | --- | ------------- | ------ | -------- | --- | -------- | ------ | ------ | ----- | ------ | ------ | --------- | --- | ---------- | ------- |
|            |     |               |        |          |     |          |        | To add | a new | accent | class, | we expand | the | classifier | head of |
mation [8], we extract hidden states from multiple Whisper thebasemodelf from|Y |outputsto|Y |+1outputs.
|     |     |     |     |     |     |     |     |     |     | base |     | old |     | old |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
encoder layers rather than using only the final layer. The classifier’s weights for the old classes are copied from
| Namely, |     | we select | encoder | layers | S ={16,20,24,28}. |     | For |           |     |           |         |     |              |     |              |
| ------- | --- | --------- | ------- | ------ | ----------------- | --- | --- | --------- | --- | --------- | ------- | --- | ------------ | --- | ------------ |
|         |     |           |         |        |                   |     |     | f , while | the | new-class | weights |     | are randomly |     | initialized. |
|         |     |           | ={h(    | ℓ)}T   |                   |     |     | b a se    |     |           |         |     |              |     |              |
eachlayerℓ∈S,letH(ℓ) denotethecorrespond- T h e expanded model, denoted by f , is then trained on the
|                 |     |        |           | t t=1 |       |                  |     |          |               |     |      |     | θ          |         |      |
| --------------- | --- | ------ | --------- | ----- | ----- | ---------------- | --- | -------- | ------------- | --- | ---- | --- | ---------- | ------- | ---- |
| ing frame-level |     | hidden | sequence. | We    | apply | a layer-specific |     |          |               |     |      |     |            |         |      |
|                 |     |        |           |       |       |                  |     | union of | the new-class |     | data | and | the replay | memory, | i.e. |
projectionheadq ℓ (·)toreducetheencoder’shiddendimension D =D ∪D .
|     |     |     |     |     |     |     |     | train | new | mem |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
and map all selected layers to the same dimensionality: a) Domain-stratified replay memory: For class-
|     |     |     | (cid:16) | (cid:17) |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | -------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
h˜(ℓ) h(ℓ) h˜(ℓ) ∈RP, incremental training, we construct a fixed replay memory
|     |     | =q  |     | ,   |     |     | (1) |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
t ℓ t t from the old accent classes. Let Y denote the old label
old
where P =256 in our implementation. Each projection head space. For each old class k ∈Y old , we select:
consistsoflayernormalizationfollowedbyalinearprojection,
|        |             |     |          |     |     |     |     |     |     | K   | =min | (cid:0) K,|Dold| | (cid:1) |     |     |
| ------ | ----------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | ---- | ---------------- | ------- | --- | --- |
| a ReLU | activation, | and | dropout. |     |     |     |     |     |     | k   |      |                  | k       |     | (6) |
Theprojectedlayerrepresentationsarethenconcatenatedat utterances, where K is the maximum number of replay sam-
| each | frame | to form a | multi-level | accent   | representation: |     |     |          |       | Dold |         |     |              |     |            |
| ---- | ----- | --------- | ----------- | -------- | --------------- | --- | --- | -------- | ----- | ---- | ------- | --- | ------------ | --- | ---------- |
|      |       |           |             |          |                 |     |     | ples per | class | and  | denotes | the | old-training |     | utterances |
|      |       |           | (cid:16)    | (cid:17) |                 |     |     |          |       |      | k       |     |              |     |            |
h˜(ℓ) ∈R|S|P. with label k. Within each class, we stratify sampling by
|     | u   | =Concat |     | ,   | u   |     | (2) |                |     |           |     |     |            |     |           |
| --- | --- | ------- | --- | --- | --- | --- | --- | -------------- | --- | --------- | --- | --- | ---------- | --- | --------- |
|     |     | t       | ℓ∈S | t   | t   |     |     |                |     |           |     |     |            |     |           |
|     |     |         |     |     |     |     |     | domain/corpus. |     | We divide | the | K k | selections | as  | evenly as |
The resulting sequence {u }T is passed to the attentive possible across the domains represented in Dold and sam-
|     |     |     |     | t t=1 |     |     |     |     |     |     |     |     |     | k   |     |
| --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
statistics pooling module [17], which computes attention- ple without replacement from each domain. If a domain
weightedmeanandstandard-deviationstatisticsovertime.The has too few utterances, the remaining samples are drawn

at random from the unused utterances of the same class. EdAcc [23], L2ARCTIC [24], ARCTIC [25], EmiliaYO-
The resulting class-specific memories are combined to form DAS [26], VoxPopuli [27], and EMIME [28]. We assign each
D = (cid:83) Dk . This yields a replay memory that is utterance to a regional accent group following our collapsed
mem k∈Yold mem
balanced across old accent classes while maintaining source accent label mapping. The full dataset includes North Ameri-
diversity within each class. can (NA), British Isles (BI), Australasian (AUS), South Asian
b) Retention-aware continual training: Let Y = Y ∪ (SA), Southeast Asian (SEA), Spanish (SPA), and Chinese
old
{y } denote the expanded label space, where y is the (CHN) accent groups.
new new
newly added accent class. Given utterance x i , the expanded All splits are constructed at the speaker level to prevent
model produces logits a i = f θ (x i ) ∈ R|Y|. The primary speaker overlap among the training, validation, and test sets.
objective is cross-entropy L CE over the expanded label space. Because several corpora contain many utterances from the
To reduce forgetting on old accents, we keep a frozen copy same speaker, we also limit speaker dominance during train-
of the base model f base and use it as a teacher on replay ing. For each accent group, we first cap each speaker at 50
memory samples. Let B mem ⊆ B denote the old-class replay utterances. We then limit the total number of utterances per
samples in the mini-batch, and let C old = |Y old |. Since the accent group to 50,000. To preserve training dataset diversity,
expanded classifier contains an additional new-class output, wesampleacrossthedatasetsavailableforeachaccentgroup.
we restrict its logits to the old label space when computing This prevents the training subset from being dominated by
the retention loss. For replay sample i, let ao i ld = a i [Y old ] a small number of frequent speakers or large corpora, while
and ab i ase = f base (x i ) ∈ RCold denote the old-class logits of keeping variation in recording conditions across datasets. The
the expanded model and the logits produced by the frozen resulting training data distribution is shown in Figure 2.
base model, respectively. We use temperature-scaled knowl-
edge distillation [19] to preserve the base model’s old-class
Dataset
prediction distribution: EdAcc GLOBE AESRC ARCTIC voxpopuli
CommonVoice UK-Ireland L2ARCTIC EmiliaYODAS EMIME
(cid:18) abase(cid:19) (cid:18) aold(cid:19)
# utt.
pbase =softmax i , pold =softmax i ,
i τ i τ North American 21,378
(7) British Isles 28,244
where τ is the distillation temperature. The retention loss is Australasian 17,244
defined as: South Asian 21,462
L =τ2 1 (cid:88) D (cid:0) pbase ∥pold(cid:1) . (8) Southeast Asian 3,338
ret |B mem | KL i i Spanish 9,349
i∈Bmem
Chinese 3,565
This loss encourages the expanded model to preserve the
frozen base model’s old-accent decision structure on replay 0 20 40 60 80 100
Dataset composition within each accent (%)
samples, while the supervised cross-entropy loss allows the
model to learn the newly added accent. Fig. 2: Training dataset composition by accent group. Bars
BecauseL ignoresthenewlyaddedclasslogit,anoldre- show source-dataset percentages, and right-side numbers indi-
ret
playsamplecanstillreceiveahighnew-classscoreevenwhen cate utterance counts.
itsold-classlogitsarepreserved.Complementarytoretention,
we add an old-to-new margin loss that directly suppresses b) Model training: We train AccentCL sequentially.
new-classoverpredictiononreplaymemorysamples.Foreach First,wetrainthebaseregionalaccentclassifierwithafrozen
replay sample, the loss encourages the true old-class logit to Whisper-Large-v3 encoder and update only the downstream
exceed the new-class logit by a margin m: projectionlayers,attentivestatisticspoolingmodule,MLP,and
classifier head. The base model is optimized with AdamW
1 (cid:88)
L old,new = |B | max(0, a i,ynew −a i,yi +m). using a learning rate of 1 × 10−3, source mean alignment
mem i∈Bmem weightβ
align
=20,andlogitadjustmenttemperatureτ =1.0.
(9)
We select the base checkpoint using validation macro-F1.
The final continual-training objective is:
We then perform continual training by adding Spanish-
L =L +β L
CL CE align align accented English as a new class. The classifier head is ex-
(10)
+β L +β L . panded by copying the old-class weights and initializing a
ret ret old,new old,new
new output unit. The expanded model is trained with new-
Here, β , β , and β control the strengths of source
align ret old,new
class data and source-stratified replay memory from the old
alignment, retention, and old-to-new margin regularization,
classes. We set the replay budget to K = 2,000 utterances
respectively.
per old accent class, corresponding to approximately 10% of
V. EXPERIMENTALSETUP the original training data. We use a learning rate of 1×10−4,
a) Datasets: We construct our accent classification β =20,retentiontemperatureτ =2,andretentionweight
align
datasetfrommultipleEnglishspeechcorpora,includingCom- β = 1.0. The margin loss uses β = 0.2 and margin
ret old,new
monVoice [11], GLOBE [20], AESRC [21], UK-Ireland [22], m=0.3.

TABLE I: Regional accent classification results on the shared 5-class regional label space. Predictions from CommonAccent
andVoxlectaremappedtothesameregionallabelspaceforevaluation.Formodelswithalargeroutputlabelspace,predictions
outside the evaluated 5-class label set are retained and counted as errors. All metrics are reported in percentages.
Method All Acc(↑) All Bal Acc(↑) All Macro-F1(↑) OOD Acc(↑) OOD Bal Acc(↑) OOD Macro-F1(↑)
CommonAccent[4] 56.0 48.3 48.6 79.0 53.6 58.2
Voxlect[5] 64.8 65.8 68.1 83.8 77.5 82.6
AccentCL base 76.0 77.1 76.9 89.7 79.6 83.0
We repeat the same procedure to add Chinese-accented VI. RESULTS
English on top of the base-plus-Spanish model. For continual
A. Robust regional accent classification
training stages, checkpoint selection uses the harmonic mean
between old-class macro-F1 and new-class F1: Table I reports the performance of the 5-class base regional
2Fold Fnew accent classifier. We evaluate each model on the full test set
H old,new = Fold ma + cr F o new+ϵ , (11) andonasource-held-outsubsettoassessgeneralizationacross
macro recording sources. On the full evaluation set, AccentCL base
which balances old-class retention and new-class learning,
achieves an accuracy of 76.0%, balanced accuracy of 77.1%,
instead of selecting models that perform well on only one
and macro-F1 of 76.9%. Compared with the mapped Voxlect
side of the incremental task.
baseline, AccentCL base improves accuracy by +11.2 points,
c) Baselines: We compare AccentCL against the two
balanced accuracy by +11.3 points, and macro-F1 by +8.8
existing accent classification systems: CommonAccent [4] is
points. These results show that the proposed regional classi-
a wav2vec2-XLSR based English accent classifier trained on
fier provides stronger overall and class-balanced performance
CommonVoice. Voxlect [5] studies several backbone models
under the shared five-class accent label space.
for English dialect classification and reports the best perfor-
On the source-held-out subset, AccentCL base achieves
mance with Whisper-Large-v3. We therefore use its Whisper-
an OOD accuracy of 89.7%, balanced accuracy of 79.6%,
Large-v3 variant as the Voxlect baseline. Since CommonAc-
and macro-F1 of 83.0%. Compared with Voxlect, AccentCL
cent,Voxlect,andAccentCLusedifferentoutputlabelspaces,
improves OOD accuracy by +5.9 points, balanced accuracy
allmodelsareevaluatedunderthesharedregionalaccentlabel
by +2.1 points, and macro-F1 by +0.4 points. Although the
mapping described in Section V-0a. Predictions outside the
OOD macro-F1 gain is modest, AccentCL maintains strong
evaluated label set are retained and counted as errors. For the
class-balanced performance while improving overall accuracy
class-incrementalexperiments,wealsoreporttheFrozenBase
on held-out recording sources.
model before adding each new class. This gives the original
performance on the old classes and is used as the reference
TABLEII:Class-incrementalaccentlearningresultsandabla-
for measuring forgetting.
tions. Old BAcc. is computed over the classes learned before
d) Evaluation metrics: We report accuracy, balanced
each incremental step. ∆ and ∆ denote old-class
avg worst
accuracy, and macro-F1 for regional accent classification.
accuracy changes relative to the old model.
Balanced accuracy is computed as the mean per-class recall.
Macro-F1 is the unweighted average of class-wise F1 scores. Method NewF1↑ OldBAcc↑ ∆avg ↑ ∆worst ↑
We report results under two evaluation settings. All evaluates Basemodel:5regionalclasses
all test utterances from our constructed benchmark whose FrozenBase – 77.1 – –
labels fall within the target regional label set. OOD evaluates Step1:5→6,addingSpanish-accentedEnglish
source-held-outgeneralizationonSpeechAccentArchive[29] Voxlect 52.7 66.0 – –
Replayonly 81.4 76.2 -1.6 -4.4(BI)
and IDEA [30], neither of which is used as a training source Replay+retention 82.8 77.1 -0.7 -3.6(BI)
for any model. Replay+margin 82.2 76.1 -1.8 -4.6(BI)
AccentCL 83.3 77.3 -0.8 -3.9(BI)
For class-incremental learning, we report new-class F1
and old-class balanced accuracy, where old classes are those Step2:6→7,addingChinese-accentedEnglish
Voxlect – – – –
learnedbeforethecurrentincrementalstep.Tomeasureforget- Replayonly 41.9 75.1 -3.7 -7.5(SPA)
ting, we compute the per-class accuracy change between the Replay+retention 48.5 75.9 -2.9 -5.7(SPA)
Replay+margin 59.9 77.1 -1.4 -3.8(NA)
expanded model and the model before adding the new accent:
AccentCL 61.8 77.6 -1.7 -4.2(NA)
δ =Accafter−Accbefore. (12)
k k k B. Class-incremental accent expansion
We then report the average forgetting over all old classes and
We next evaluate whether the regional accent inventory can
the worst-class change:
be expanded without retraining the full model from scratch.
∆ = 1 (cid:88) min(0,δ ), ∆ = min δ . Starting from the five-class base model, we incrementally
avg |Y old |
k∈Yold
k worst k∈Yold k add two accent classes: first Spanish-accented English, and
(13) then Chinese-accented English, yielding a class-incremental
More negative values indicate stronger forgetting. sequence of 5→6→7 classes.

a) Incremental performance: Table II reports the class-
incremental results. Voxlect is included as a fixed-label ref-
erence model rather than a continual-learning baseline. After
adding Spanish-accented English, AccentCL achieves the best
new-classF1(83.3)andold-classbalancedaccuracy(77.3%),
while showing only a small average old-class drop (-0.8
points).Theretention-onlyvariantgivesthesmallestforgetting
in this step, suggesting that matching the previous classifier’s
predictions helps preserve the performance of the original
five regional classes. After adding Chinese-accented English,
AccentCL again obtains the best new-class F1 (61.8) and old-
classbalancedaccuracy(77.6%).Themargin-onlyvarianthas
the smallest degradation on old classes in this step, indicating
that reducing confusions between old accents and the new
class is important when adding Chinese-accented English.
Overall, the ablations show that retention and the margin loss
contribute to the two expansion steps. Combining both gives
ACCENTCL the most consistent balance between learning the
new accent and maintaining performance on old accents.
b) Qualitative analysis: Figure 3 compares per-class
recallforVoxlectandAccentCLafteraddingSpanish-accented
English. We evaluate on the held-out speakers from corpora
used by both models. AccentCL shows higher recall for
the newly added Spanish-accented English class and for old
regionalclasses,whichisconsistentwiththegainsinTableII.
Fig. 4 shows t-SNE visualizations of learned embedding
spaces. Compared with Voxlect, AccentCL forms clearer ac-
cent groups with weaker domain shifts. For example, Aus-
tralasiansamplesinVoxLectaresplitintoseveralclustersfrom
different domains, while AccentCL keeps them more com-
pact. Some domain shifts remain, suggesting that AccentCL
reduces, but does not fully remove, variation across corpora.
VII. DISCUSSION
We proposed ACCENTCL, a framework for English ac-
cent classification that supports class-incremental expansion.
By combining data from new accents with replay memory
stratified by dataset, AccentCL adds Spanish- and Chinese-
accented English without retraining on the full old dataset. In
oursetting,onlyabout10%oftheoldtrainingdataisretained
NA
BI
AUS
SA
SEA
SPA
N
A BI
A
US
S
A
SE
A SPA Out
lebal
eurT
t-SNE 1
(a) Voxlect (b) AccentCL
0.670.08 0.07 0.06 0.740.09 0.050.07
0.80 0.08 0.85
0.110.75 0.08 0.090.86
0.93 0.93
0.120.68 0.10 0.08 0.84
0.210.17 0.08 0.380.11 0.110.21 0.060.61
N
A BI
A
US
S
A
SE
A SPA
Predicted label
Fig. 3: Per-class recall matrices for Voxlect and AccentCL af-
ter adding Spanish-accented English. Out denotes predictions
outside the evaluated label set.
2
ENS-t
(I) Colored by accent
t-SNE 1
2
ENS-t
(II) Colored by data source
(a) Voxlect.
t-SNE 1
2
ENS-t
(I) Colored by accent
t-SNE 1
Australasian (1000) South Asian (1000)
British Isles (1000) Southeast Asian (402)
North American (1000) Spanish (1000)
2
ENS-t
(II) Colored by data source
Aesrc (138) Idea (1834)
Commonvoice (1489) L2Arctic (14)
Edacc (26) Speechaccentarchive (467)
Emiliayodas (246) Uk Ireland (48)
Globe (1124) Voxpopuli (16)
(b) AccentCL.
Fig. 4: t-SNE of Voxlect and AccentCL embeddings after
addingSpanish-accentedEnglish.Pointsarecoloredbyaccent
label (left) and by domain (right).
asreplaymemoryandstillmaintainsbalancedaccuracyonold
accent classes across incremental steps.
The ablation results show that retention and old-to-new
margin regularization provide complementary benefits. Re-
tention preserves the previous classifier’s behavior while the
margin loss reduces confusion between old accents and the
newly added class. New-class F1, balanced accuracy on old
classes,andworst-classforgettingfurthershowthatAccentCL
balances learning new accents with retaining old accents.
Compared with the Voxlect state-of-the-art baseline, Ac-
centCL achieves stronger regional accent classification and
better adaptation to the expanded label space. These results
show that AccentCL provides a practical pipeline for continu-
ally expanding accent classifiers to future accent categories
without retraining from scratch or substantially degrading
previously learned classes.
ACKNOWLEDGMENTS
Generative AI tools were used only for editing to improve
clarity and presentation. All experimental design, implemen-
tation, data analysis, and result interpretation were performed
by the authors.

REFERENCES
[21] X.Shi,F.Yu,Y.Lu,Y.Liang,Q.Feng,D.Wang,Y.Qian,andL.Xie,
“Theaccentedenglishspeechrecognitionchallenge2020:opendatasets,
|     |     |     |     |     |     |     |     | tracks, | baselines, | results | and methods,” | in ICASSP | 2021-2021 | IEEE |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ---------- | ------- | ------------- | --------- | --------- | ---- |
[1] X.Zhang,X.Zhang,K.Peng,Z.Tang,V.Manohar,Y.Liu,J.Hwang,
D.Li,Y.Wang,J.Chan,Y.Huang,Z.Wu,andM.Ma,“Vevo:Control- International Conference on Acoustics, Speech and Signal Processing
lablezero-shotvoiceimitationwithself-superviseddisentanglement,”in (ICASSP). IEEE,2021,pp.6918–6922.
[22] I.Demirsahin,O.Kjartansson,A.Gutkin,andC.Rivera,“Open-source
| ICLR.          | OpenReview.net,2025. |                |        |               |        |                     |                |                |     |            |               |           |        |                 |
| -------------- | -------------------- | -------------- | ------ | ------------- | ------ | ------------------- | -------------- | -------------- | --- | ---------- | ------------- | --------- | ------ | --------------- |
|                |                      |                |        |               |        |                     |                | Multi-speaker  |     | Corpora of | the English   | Accents   | in the | British Isles,” |
| [2] W. Quamer, |                      | M.-R.          | Tseng, | G. Nasrallah, | and    | R. Gutierrez-Osuna, |                |                |     |            |               |           |        |                 |
|                |                      |                |        |               |        |                     |                | in Proceedings |     | of The     | 12th Language | Resources |        | and Evaluation  |
| “Phonos:       | Phonetic             | neutralization |        | for           | online | streaming           | applications,” |                |     |            |               |           |        |                 |
arXivpreprintarXiv:2603.27001,2026. Conference(LREC). Marseille,France:EuropeanLanguageResources
[3] K.Prinos,N.Patwari,andC.A.Power,“Speakingofaccent:Acontent Association (ELRA), May 2020, pp. 6532–6541. [Online]. Available:
https://www.aclweb.org/anthology/2020.lrec-1.804
analysisofaccentmisconceptionsinasrresearch,”inProceedingsofthe
|      |                |     |              |     |                 |     |               | [23] R. | Sanabria, | N. Bogoychev,  | N.            | Markl, A. | Carmantini, | O. Klejch,      |
| ---- | -------------- | --- | ------------ | --- | --------------- | --- | ------------- | ------- | --------- | -------------- | ------------- | --------- | ----------- | --------------- |
| 2024 | ACM Conference |     | on Fairness, |     | Accountability, | and | Transparency, |         |           |                |               |           |             |                 |
|      |                |     |              |     |                 |     |               | and     | P. Bell,  | “The edinburgh | international | accents   | of          | english corpus: |
2024,pp.1245–1254.
Towardsthedemocratizationofenglishasr,”inICASSP2023-2023IEEE
| [4] J. Zuluaga-Gomez, |     |     | S. Ahmed, | D.  | Visockas, | and | C. Subakan, |     |     |     |     |     |     |     |
| --------------------- | --- | --- | --------- | --- | --------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
“Commonaccent:Exploringlargeacousticpretrainedmodelsforaccent International Conference on Acoustics, Speech and Signal Processing
|                |     |       |           |         |             |     |             | (ICASSP).     |     | IEEE,2023,pp.1–5. |            |              |                       |     |
| -------------- | --- | ----- | --------- | ------- | ----------- | --- | ----------- | ------------- | --- | ----------------- | ---------- | ------------ | --------------------- | --- |
| classification |     | based | on common | voice,” | Interspeech |     | 2023, 2023. |               |     |                   |            |              |                       |     |
|                |     |       |           |         |             |     |             | [24] G. Zhao, | S.  | Sonsaat, A.       | Silpachai, | I. Lucic, E. | Chukharev-Hudilainen, |     |
[Online].Available:https://arxiv.org/abs/2305.18283
|              |     |        |        |         |                  |     |              | J. Levis, | and | R. Gutierrez-Osuna, |     | “L2-arctic: | A non-native | english |
| ------------ | --- | ------ | ------ | ------- | ---------------- | --- | ------------ | --------- | --- | ------------------- | --- | ----------- | ------------ | ------- |
| [5] T. Feng, | K.  | Huang, | A. Xu, | X. Shi, | T. Lertpetchpun, | J.  | Lee, Y. Lee, |           |     |                     |     |             |              |         |
speechcorpus,”inProc.Interspeech2018,2018,pp.2783–2787.
| D. Byrd, | and | S. Narayanan, |     | “Voxlect: | A speech | foundation | model |     |     |     |     |     |     |     |
| -------- | --- | ------------- | --- | --------- | -------- | ---------- | ----- | --- | --- | --- | --- | --- | --- | --- |
benchmark for modeling dialects and regional languages around the [25] J. Kominek and A. W. Black, “The cmu arctic speech databases,” in
Proc.SSW2004,2004,pp.223–224.
| globe,” | in Proceedings |     | of the | 32nd | ACM | SIGKDD Conference |     | on  |     |     |     |     |     |     |
| ------- | -------------- | --- | ------ | ---- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- |
[26] H.He,Z.Shang,C.Wang,X.Li,Y.Gu,H.Hua,L.Liu,C.Yang,J.Li,
KnowledgeDiscoveryandDataMiningV.1,2026,pp.2640–2651.
P.Shietal.,“Emilia:Alarge-scale,extensive,multilingual,anddiverse
| [6] A. Baevski, |     | Y. Zhou, | A. Mohamed, |     | and | M. Auli, “wav2vec | 2.0: |     |     |     |     |     |     |     |
| --------------- | --- | -------- | ----------- | --- | --- | ----------------- | ---- | --- | --- | --- | --- | --- | --- | --- |
datasetforspeechgeneration,”IEEETransactionsonAudio,Speechand
| A framework |     | for self-supervised |     | learning | of  | speech representations,” |     |     |     |     |     |     |     |     |
| ----------- | --- | ------------------- | --- | -------- | --- | ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
LanguageProcessing,2025.
Advancesinneuralinformationprocessingsystems,vol.33,pp.12449– [27] C. Wang, M. Riviere, A. Lee, A. Wu, C. Talnikar, D. Haziza,
12460,2020.
|              |     |       |             |     |           |        |           | M.  | Williamson, | J. Pino, | and E. | Dupoux, “Voxpopuli: |     | A large-scale |
| ------------ | --- | ----- | ----------- | --- | --------- | ------ | --------- | --- | ----------- | -------- | ------ | ------------------- | --- | ------------- |
| [7] A. Babu, | C.  | Wang, | A. Tjandra, | K.  | Lakhotia, | Q. Xu, | N. Goyal, |     |             |          |        |                     |     |               |
multilingualspeechcorpusforrepresentationlearning,semi-supervised
K.Singh,P.VonPlaten,Y.Saraf,J.Pinoetal.,“Xls-r:Self-supervised
learningandinterpretation,”inProceedingsofthe59thAnnualMeeting
| cross-lingual |     | speech | representation | learning |     | at scale,” | arXiv preprint |     |     |     |     |     |     |     |
| ------------- | --- | ------ | -------------- | -------- | --- | ---------- | -------------- | --- | --- | --- | --- | --- | --- | --- |
oftheAssociationforComputationalLinguisticsandthe11thInterna-
arXiv:2111.09296,2021. tional Joint Conference on Natural Language Processing (Volume 1:
[8] S.Chen,C.Wang,Z.Chen,Y.Wu,S.Liu,Z.Chen,J.Li,N.Kanda, LongPapers),2021,pp.993–1003.
| T. Yoshioka, |     | X. Xiao    | et al., | “Wavlm:      | Large-scale | self-supervised | pre-        |         |              |       |           |            |                |          |
| ------------ | --- | ---------- | ------- | ------------ | ----------- | --------------- | ----------- | ------- | ------------ | ----- | --------- | ---------- | -------------- | -------- |
|              |     |            |         |              |             |                 |             | [28] M. | Wester, “The | emime | bilingual | database,” | The University | of Edin- |
| training     | for | full stack | speech  | processing,” | IEEE        | Journal         | of Selected |         |              |       |           |            |                |          |
burgh,Tech.Rep.,2010.
TopicsinSignalProcessing,vol.16,no.6,pp.1505–1518,2022.
[29] S.H.WeinbergerandS.A.Kunath,“Thespeechaccentarchive:towards
[9] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, atypologyofenglishaccents.”Language&Computers,vol.73,no.1,
| and I. | Sutskever, | “Robust |     | speech | recognition | via large-scale | weak | 2011. |     |     |     |     |     |     |
| ------ | ---------- | ------- | --- | ------ | ----------- | --------------- | ---- | ----- | --- | --- | --- | --- | --- | --- |
supervision,”2022.[Online].Available:https://arxiv.org/abs/2212.04356
[30] P.MeierandS.Muller,“Idea:Internationaldialectsofenglisharchive,”
[10] M.Yang,R.C.M.C.Shekar,O.Kang,andJ.H.L.Hansen,“WhatCan
AccessedMay,vol.17,p.2005,1998.
anAccentIdentifierLearn?ProbingPhoneticandProsodicInformation
inaWav2vec2-basedAccentIdentificationModel,”inInterspeech2023,
2023,pp.1923–1927.
| [11] R. Ardila,        | M.  | Branson,  | K. Davis, | M.       | Kohler,        | J. Meyer, | M. Henretty,   |     |     |     |     |     |     |     |
| ---------------------- | --- | --------- | --------- | -------- | -------------- | --------- | -------------- | --- | --- | --- | --- | --- | --- | --- |
| R. Morais,             | L.  | Saunders, | F. Tyers, | and      | G. Weber,      | “Common   | voice:         | A   |     |     |     |     |     |     |
| massively-multilingual |     |           | speech    | corpus,” | in Proceedings |           | of the twelfth |     |     |     |     |     |     |     |
languageresourcesandevaluationconference,2020,pp.4218–4222.
| [12] G. M. | Van de | Ven | and A. | S. Tolias, | “Three | scenarios | for continual |     |     |     |     |     |     |     |
| ---------- | ------ | --- | ------ | ---------- | ------ | --------- | ------------- | --- | --- | --- | --- | --- | --- | --- |
learning,”arXivpreprintarXiv:1904.07734,2019.
| [13] J. Kirkpatrick, |             | R. Pascanu,  |              | N. Rabinowitz, | J.        | Veness,              | G. Desjardins, |     |     |     |     |     |     |     |
| -------------------- | ----------- | ------------ | ------------ | -------------- | --------- | -------------------- | -------------- | --- | --- | --- | --- | --- | --- | --- |
| A. A.                | Rusu,       | K. Milan,    | J. Quan,     | T.             | Ramalho,  | A. Grabska-Barwinska |                |     |     |     |     |     |     |     |
| et al.,              | “Overcoming |              | catastrophic | forgetting     | in        | neural networks,”    | Pro-           |     |     |     |     |     |     |     |
| ceedings             | of          | the national | academy      | of             | sciences, | vol. 114,            | no. 13, pp.    |     |     |     |     |     |     |     |
3521–3526,2017.
[14] Z.LiandD.Hoiem,“Learningwithoutforgetting,”IEEEtransactions
onpatternanalysisandmachineintelligence,vol.40,no.12,pp.2935–
2947,2017.
| [15] S.-A. | Rebuffi, | A. Kolesnikov, |     | G. Sperl, | and | C. H. Lampert, | “icarl: |     |     |     |     |     |     |     |
| ---------- | -------- | -------------- | --- | --------- | --- | -------------- | ------- | --- | --- | --- | --- | --- | --- | --- |
Incrementalclassifierandrepresentationlearning,”inProceedingsofthe
IEEEconferenceonComputerVisionandPatternRecognition,2017,pp.
2001–2010.
| [16] G. Li,          | W. Yu, | Y. Yao,   | W.  | Tong,         | Y. Liang, | Q. Lin,       | and T. Yang, |     |     |     |     |     |     |     |
| -------------------- | ------ | --------- | --- | ------------- | --------- | ------------- | ------------ | --- | --- | --- | --- | --- | --- | --- |
| “A retention-centric |        | framework |     | for continual |           | learning with | guaranteed   |     |     |     |     |     |     |     |
modeldevelopmentalsafety,”arXivpreprintarXiv:2410.03955,2024.
[17] K.Okabe,T.Koshinaka,andK.Shinoda,“AttentiveStatisticsPooling
| for Deep | Speaker | Embedding,” |     | in Interspeech |     | 2018, 2018, | pp. 2252– |     |     |     |     |     |     |     |
| -------- | ------- | ----------- | --- | -------------- | --- | ----------- | --------- | --- | --- | --- | --- | --- | --- | --- |
2256.
| [18] A. K. | Menon,     | S. Jayasumana, |          | A.  | S. Rawat,          | H. Jain, | A. Veit, and   |     |     |     |     |     |     |     |
| ---------- | ---------- | -------------- | -------- | --- | ------------------ | -------- | -------------- | --- | --- | --- | --- | --- | --- | --- |
| S. Kumar,  | “Long-tail |                | learning | via | logit adjustment,” |          | arXiv preprint |     |     |     |     |     |     |     |
arXiv:2007.07314,2020.
[19] G.Hinton,O.Vinyals,andJ.Dean,“Distillingtheknowledgeinaneural
network,”arXivpreprintarXiv:1503.02531,2015.
[20] W.Wang,Y.Song,andS.Jha,“GLOBE:AHigh-qualityEnglishCorpus
| with | Global | Accents | for Zero-shot | Speaker | Adaptive | Text-to-Speech,” |     |     |     |     |     |     |     |     |
| ---- | ------ | ------- | ------------- | ------- | -------- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- |
inInterspeech2024,2024,pp.1365–1369.