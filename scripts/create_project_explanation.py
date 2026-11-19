"""Explain the saved MIA/DP experiments; never execute experiment cells."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/experiment_3_and_4'
PROPOSAL = ROOT.parent / 'proposal-template/Project_Proposal_Syed.pdf'
E3 = ROOT / 'artifacts/exp_3_2023_2014_100000_nominal_paper_algorithms/reduced_k6_s16_m7'
E4 = ROOT / 'artifacts/exp_4_2023_2014_dp_accuracy_e0p5_1_2_5_10_25_50_d1e6_mixed_mlp'
NAME = 'Syed Khaja Ikramuddin'
FONT = 'Times New Roman'
plt.rcParams.update({'font.family': FONT, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def percent(value, precision=2):
    return f'{100 * float(value):.{precision}f}'


class Sheet:
    """Plain report renderer, kept with the current report generator."""
    def __init__(self, pdf, number, title, subtitle):
        self.pdf, self.number = pdf, number
        self.fig = plt.figure(figsize=(8.27, 11.69), dpi=120, facecolor='white')
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 1), ylim=(0, 1))
        self.ax.axis('off')
        self.renderer = self.fig.canvas.get_renderer()
        self.text(.08, .949, title, 16, bold=True)
        self.y = .914
        self.para(subtitle, size=10.3, gap=.012)
        self.text(.08, .04, f'{NAME} | ECE750T | Project notes, 18 November 2026', 8.3, color='.35')
        self.text(.92, .04, str(number), 9, ha='right')

    def text(self, x, y, text, size, bold=False, color='black', **kwargs):
        self.ax.text(x, y, text, fontsize=size, fontweight='bold' if bold else 'normal',
                     color=color, va='top', transform=self.ax.transAxes, **kwargs)

    def wrap(self, text, width, size, bold=False):
        font = FontProperties(family=FONT, size=size, weight='bold' if bold else 'normal')
        limit = width * self.fig.get_figwidth() * self.fig.dpi
        output = []
        for paragraph in str(text).split('\n'):
            line = ''
            for word in paragraph.split():
                candidate = f'{line} {word}'.strip()
                if self.renderer.get_text_width_height_descent(candidate, font, False)[0] <= limit:
                    line = candidate
                else:
                    if line:
                        output.append(line)
                    line = word
            output.append(line)
        return output

    def check(self):
        if self.y < .072:
            raise ValueError(f'Page {self.number} is too long: bottom {self.y:.3f}')

    def para(self, text, size=11.2, gap=.008, bold=False):
        lines = self.wrap(text, .84, size, bold)
        step = size * 1.22 / 72 / self.fig.get_figheight()
        for line in lines:
            self.text(.08, self.y, line, size, bold)
            self.y -= step
        self.y -= gap
        self.check()

    def heading(self, text):
        self.y -= .003
        self.para(text, size=12, bold=True, gap=.003)

    def table(self, headers, rows, proportions):
        widths = .84 * np.asarray(proportions) / sum(proportions)
        xs = .08 + np.r_[0, widths.cumsum()[:-1]]
        for index, row in enumerate([headers]+rows):
            head = index == 0
            size = 10.2
            lines = [self.wrap(cell, width-.012, size, head) for cell,width in zip(row,widths)]
            step = size * 1.12 / 72 / self.fig.get_figheight()
            height = max(map(len,lines))*step + .009
            if head:
                self.ax.add_patch(Rectangle((.08,self.y-height),.84,height,facecolor='.94',edgecolor='none'))
            for x, cell in zip(xs,lines):
                for j,line in enumerate(cell):
                    self.text(x+.006,self.y-.004-j*step,line,size,head)
            self.y -= height
            self.ax.plot([.08,.92],[self.y,self.y],color='.65' if head else '.85',linewidth=.5)
            self.check()
        self.y -= .009


class ExplanationSheet(Sheet):
    def box(self, x, top, width, height, label, size=10.1):
        lines=self.wrap(label,width-.016,size)
        step=size*1.14/72/self.fig.get_figheight()
        if len(lines)*step > height-.008:
            raise ValueError(f'Pipeline box too small: {label}')
        self.ax.add_patch(Rectangle((x,top-height),width,height,facecolor='white',edgecolor='.3',linewidth=.7))
        y=top-(height-len(lines)*step)/2
        for line in lines:
            self.text(x+width/2,y,line,size,ha='center')
            y-=step

    def arrow(self, start, end):
        self.ax.add_patch(FancyArrowPatch(start,end,arrowstyle='->',mutation_scale=9,
                                        color='.2',linewidth=.8))

    def pipeline(self, labels):
        """Six steps in two rows, with arrows that follow the actual order."""
        assert len(labels)==6
        top,width,height,gap=self.y-.003,.25,.058,.030
        xs=[.08,.375,.67]
        for i,label in enumerate(labels):
            row=i//3
            column=i%3 if row==0 else 2-i%3
            x,y=xs[column],top-row*(height+gap)
            self.box(x,y,width,height,f'{i+1}. {label}')
        for x in xs[:-1]:
            self.arrow((x+width+.003,top-height/2),(x+.292,top-height/2))
        self.arrow((.795,top-height-.002),(.795,top-height-gap+.004))
        for column in [2,1]:
            self.arrow((xs[column]-.003,top-height-gap-height/2),
                       (xs[column-1]+width+.003,top-height-gap-height/2))
        self.y=top-2*height-gap-.012
        self.check()

    def branches(self, left, right, final):
        """Independent target and shadow preparation converge on evaluation."""
        assert len(left)==len(right)==3
        top,width,height,gap=self.y-.003,.38,.049,.018
        for x,labels in [(.08,left),(.54,right)]:
            for i,label in enumerate(labels):
                node_top=top-i*(height+gap)
                self.box(x,node_top,width,height,label)
                if i<2:
                    self.arrow((x+width/2,node_top-height-.002),
                               (x+width/2,node_top-height-gap+.004))
        bottom=top-3*height-2*gap
        end_top=bottom-.030
        self.box(.23,end_top,.54,.049,final)
        self.arrow((.27,bottom-.002),(.39,end_top+.004))
        self.arrow((.73,bottom-.002),(.61,end_top+.004))
        self.y=end_top-.049-.012
        self.check()

    def finish(self):
        self.fig.canvas.draw()
        width,height=self.fig.canvas.get_width_height()
        for artist in self.ax.texts:
            box=artist.get_window_extent(self.fig.canvas.get_renderer())
            if box.x0<0 or box.y0<0 or box.x1>width+1 or box.y1>height+1:
                raise ValueError(f'Clipped text on page {self.number}: {artist.get_text()}')
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def build_explanation():
    OUT.mkdir(parents=True,exist_ok=True)
    proposal='\n'.join(p.extract_text() for p in PdfReader(PROPOSAL).pages)
    assert 'Uncertainty-Aware Membership Inference and Differential Privacy' in proposal
    s3=json.loads((E3/'summary.json').read_text(encoding='utf-8'))
    p3=json.loads((E3/'uncertainty_mf_mia/protocol.json').read_text(encoding='utf-8'))
    p4=json.loads((E4/'protocol.json').read_text(encoding='utf-8'))
    pfv=pd.read_csv(E3/'missing_feature_pfv/comparison.csv')
    conditional=pd.read_csv(E3/'uncertainty_mf_mia/comparison.csv')
    conditional=conditional.loc[conditional.dataset=='2023_target']
    dp=pd.read_csv(E4/'dp_target_and_mia_metrics.csv')
    assert p3['k']==6 and len(p3['masks'])==7 and len(conditional)==28 and len(dp)==8
    assert p4['epsilon_budgets']==[.5,1,2,5,10,25,50] and p4['delta']==1e-6
    m=s3['target_mia_metrics']
    baseline_table=[['Auxiliary attack test',percent(s3['shadow_attack_metrics']['test']['accuracy'],3),
                     f'{s3["shadow_attack_metrics"]["test"]["roc_auc"]:.3f}',
                     percent(s3['shadow_attack_metrics']['test']['tpr_at_1pct_fpr_roc'])],
                    ['Target MIA: complete features',percent(m['accuracy'],3),f'{m["roc_auc"]:.3f}',percent(m['tpr_at_1pct_fpr_roc'])]]
    pfv_table=[[r.condition,percent(r.accuracy,3),f'{r.roc_auc:.3f}',percent(r.tpr_at_1pct_fpr_roc)]
               for r in pfv.itertuples()]
    cond_table=[]
    for method,label,inputs in [('single','One conditional draw','First score'),
                                ('average','Average score','Mean μ'),
                                ('learned_mean','Mean-only RF','μ'),
                                ('uncertainty','Uncertainty RF','[μ, σ]')]:
        row=conditional.loc[conditional.method==method,['accuracy','roc_auc','tpr_at_1pct_fpr_roc']].mean()
        cond_table.append([label,inputs,percent(row.accuracy,3),f'{row.roc_auc:.3f}',percent(row.tpr_at_1pct_fpr_roc)])
    dp_table=[[f'{r.epsilon_requested:g}' if r.private else 'Non-private (∞)',percent(r.target_test_accuracy),
               percent(r.mia_accuracy,3),f'{r.mia_roc_auc:.3f}',percent(r.mia_tpr_at_1pct_fpr_roc)] for r in dp.itertuples()]
    example_scores=[np.array([.48,.49,.50,.50,.51,.52]),np.array([.20,.20,.20,.80,.80,.80])]
    example_table=[[name,', '.join(f'{x:.2f}' for x in scores),f'{scores.mean():.3f}',f'{scores.std(ddof=0):.3f}']
                   for name,scores in zip(['A','B'],example_scores)]

    pages=[
        ('Project explanation: MIA and differential privacy',
         f'{NAME} | ECE750T, University of Waterloo | 18 November 2026',[
            ('heading','Problem and connection to my proposal'),
            ('para','My proposal asks whether uncertainty in reconstructed missing features improves membership inference, '
             'and how differential privacy changes the privacy–utility trade-off. Hiding an attribute may not hide '
             'whether a record was used for training: related fields and model responses may still reveal membership. '
             'I want to test whether variation across plausible completions adds information beyond one completion or its average score.'),
            ('para','The experiments below use California ACS 2023 for the target and ACS 2014 as auxiliary data. '
             'The task is nominal income > $50,000 from 48 fields. This adapts the idea in my proposal, which '
             'originally named Adult and GSS. I separate the reasoning behind each method from the pipeline '
             'actually implemented in the notebooks.'),
            ('heading','Black-box setting and assumptions'),
            ('para','• API access: the attacker knows how to call the target API, its 48-field input schema/codes '
             'and class order. It receives both income-class probabilities. Training records, model weights, '
             'gradients and private preprocessing statistics are unavailable to the attacker. The experiment '
             'simulates this interface with a local probability-query wrapper.'),
            ('para','• Candidate knowledge: observed fields, the names of the hidden fields and the true income '
             'class are known. The hidden values and training membership are unknown. Income labels are '
             'used for attack features such as loss, not to condition reconstruction.'),
            ('para','• Auxiliary access: the attacker has 2014 data and trains its own shadow models, so it knows '
             'their IN/OUT membership labels. It does not need the target architecture or exact training settings. '
             'The researcher knows target membership only for evaluation; those labels do not fit the attack.'),
            ('para','• Query allowance: the API accepts completed and perturbed records repeatedly. Fees, rate '
             'limits, output rounding and plausibility filters are not simulated. The target still expects all '
             '48 fields, so the attacker must fill missing inputs before querying it.'),
            ('heading','Data splits and what the metrics mean'),
            ('table',(['Use','Records'],[
                ['Target training','100,000'],['Target validation / utility test','10,000 / 10,000'],
                ['Target membership evaluation','10,000 train-subset members + 10,000 separate nonmembers']],[.40,.60])),
            ('para','MIA accuracy is the fraction of correct member/nonmember decisions. ROC-AUC measures score '
             'ranking across cutoffs. TPR@1%FPR measures member detection while allowing at most 1% empirical '
             'false positives on the evaluation ROC. These are attack metrics; income-prediction test accuracy measures target utility.'),
            ('small','The balanced target membership set has 50% chance accuracy. AUC 0.5 is random ranking. '
             'The results are saved observations, not new runs. Source notebooks: '
             'exp_3_expanded_missing_feature_baseline.ipynb and exp_4_differential_privacy_accuracy.ipynb.')
        ]),
        ('1. MIA baseline: complete features',
         'Experiment 3 · the reference attack before any features are hidden',[
            ('heading','Methodology: why membership may leave a signal'),
            ('para','A model may behave differently on records it trained on, for example through lower loss or '
             'more stable predictions. Shadow models provide examples of these differences because their '
             'training membership is known. A separate membership classifier learns the relationship '
             'between response features and IN/OUT labels, then transfers it to the target. This is a '
             'hypothesis to test; high confidence alone does not prove membership.'),
            ('heading','Pipeline in the experiment'),
            ('branches',([
                '2023 ACS → fixed target partitions',
                'Fit train-only preprocessing + MLP\n512/256/128; 100 fixed epochs',
                'Query 20,000 target candidates\nand extract four attack signals'],[
                '2014 ACS → 16 seeded shadow splits\n40% train / 20% val / 40% OUT',
                'Fit 6 RF + 5 gradient boosting\n+ 5 XGBoost shadows; train-only preprocessing',
                'Score 2k IN + 2k OUT per shadow;\nfit RF attack; select validation cutoffs'],
                'Apply the frozen attack to target features\n→ membership scores and final decisions')),
            ('para','Repeated auxiliary IDs stay in one grouped attack train/validation/test role. The four '
             'features are the true-class log-probability ratio, entropy, cross-entropy loss and mean '
             'blind-walk step at which the prediction first changes. The target is evaluated at its '
             'final epoch; validation does not select its weights in this Exp 3 configuration.'),
            ('heading','Illustrative example — these are not measured record scores'),
            ('para','Suppose the known income class is >$50,000 and the target returns [0.12, 0.88]. '
             'The attack computes log(0.88/0.12), entropy and loss −log(0.88). If two blind walks '
             'first change the predicted class at steps 4 and 7, the walk feature is 5.5. The RF '
             'turns these features into a separate membership score. For example, a hypothetical '
             'score of 0.63 exceeds the saved accuracy cutoff 0.507089, producing a “member” decision. '
             'The income probability 0.88 is not itself a membership probability.'),
            ('heading','Saved results'),
            ('table',(['Evaluation','MIA accuracy %','AUC','TPR@1%FPR %'],baseline_table,[.43,.19,.16,.22])),
            ('small','Target income accuracy: 99.849% train and 79.52% test. The target MIA accuracy '
             'cutoff gives TPR 98.01% but FPR 72.76%. Its ROC TPR at 1% FPR is only 0.17%; '
             'overall accuracy therefore does not describe a strong low-false-alarm attack.')
        ]),
        ('2. Missing-feature PFV baseline',
         'Experiment 3 · one unconditional fill, followed by the same complete-feature attack',[
            ('heading','Methodology: fill hidden fields from auxiliary frequencies'),
            ('para','PFV uses the empirical 2014 distribution of each hidden field. It draws one observed '
             'value for each field independently, without using the visible fields to choose that value. '
             'This is a simple reconstruction baseline: it allows a complete input to be queried, but '
             'does not model whether that fill is plausible for this particular person.'),
            ('heading','Pipeline in the experiment'),
            ('pipeline',[
                'Choose one of seven\npredeclared feature masks',
                'Remove the true values\nof the hidden fields',
                'Draw one 2014 marginal\nvalue for each hidden field',
                'Query the fixed target\nwith all 48 fields filled',
                'Extract the same four\nsignals, including walks',
                'Apply the same frozen\nRF attack and cutoffs']),
            ('para','The target and attack classifier are not retrained for a mask. Observed values remain '
             'unchanged, naturally missing auxiliary donor values are excluded, and the sampling does '
             'not use income or membership labels. The same 20,000 candidates are used for every mask.'),
            ('heading','Illustrative example'),
            ('para','Suppose a candidate’s age is 35 and hours worked per week (WKHP) are hidden. PFV might '
             'sample 20 hours from the overall 2014 frequency distribution, regardless of that candidate’s '
             'age, education or job. The completed record goes through the target and frozen membership '
             'attack. The actual hidden WKHP value is never used. With two missing fields, PFV draws '
             'each independently.'),
            ('heading','Saved results for all seven masks'),
            ('table',(['Hidden fields','MIA accuracy %','AUC','TPR@1%FPR %'],pfv_table,[.43,.19,.16,.22])),
            ('small','MAR = marital status; WKHP = hours worked/week; COW = worker class; SEX = sex code. '
             '“complete” is the unmasked baseline. These are single-run results, not averages over seeds.'),
            ('para','All seven masks reduce accuracy and AUC relative to complete features in this run. '
             'Hiding all four fields still leaves 57.805% MIA accuracy, so masking does not eliminate '
             'the observed signal. Some low-FPR figures increase, which shows why accuracy and '
             'low-false-alarm performance need to be read separately.')
        ]),
        ('3. Uncertainty-aware MF-MIA: study protocol',
         'Experiment 3 · test the information in score variability, using identical conditional draws',[
            ('heading','Methodology: compare mean score with mean and variability'),
            ('para','Conditional reconstruction uses visible fields to generate several plausible fills. '
             'For each candidate I obtain K=6 membership scores and compute their mean μ and '
             'population standard deviation σ. The proposal’s hypothesis is that σ may add information '
             'about membership beyond μ. A matched RF(μ) control is needed to test the added value '
             'of RF(μ, σ); high σ does not automatically mean “member”.'),
            ('heading','Pipeline in the experiment'),
            ('pipeline',[
                'Fit conditional forests\non reserved 2014 records',
                'Hide selected fields;\nsample six completed records',
                'Query each completion;\nget six frozen-base-RF scores',
                'Summarize first score,\nmean μ and SD σ',
                'Fit on auxiliary meta IDs;\ncalibrate each method separately',
                'Evaluate auxiliary test,\nthen the fixed target candidates']),
            ('small','Reconstruction IDs exclude every attack-candidate ID. Base-attack validation IDs are '
             'split into meta-fit and calibration groups; auxiliary test IDs remain held out. The same '
             'draws support all four controls. No target membership labels choose thresholds or fit a model.'),
            ('heading','Illustrative example: same mean, different uncertainty'),
            ('para','For MAR+COW, the chain samples MAR from visible fields, then COW from visible fields '
             'plus that sampled MAR; it repeats this procedure six times. The hypothetical score sets '
             'below show why averaging and uncertainty-aware inference can make different decisions:'),
            ('table',(['Candidate','Six membership scores (illustrative)','Mean μ','SD σ'],example_table,[.13,.59,.14,.14])),
            ('small','An average-score rule treats A and B equally. RF(μ, σ) could distinguish them only '
             'if the shadow data support an association between variability and membership. These '
             'example scores and SDs are explanatory calculations, not experimental observations.'),
            ('heading','Saved target results: equal-weight averages across seven masks'),
            ('table',(['Decision method','Input','Accuracy %','AUC','TPR@1%FPR %'],cond_table,[.30,.15,.19,.14,.22])),
            ('para','Averaging improves over a single conditional draw. The uncertainty RF does not show '
             'an overall gain over the matched mean RF, and both learned RFs transfer poorly compared '
             'with averaging. This is a negative result for the proposed improvement so far; it does '
             'not imply that uncertainty must help or that higher σ has a fixed membership meaning.')
        ]),
        ('4. DP analysis: utility and membership leakage',
         'Experiment 4 · exp_4_differential_privacy_accuracy.ipynb',[
            ('heading','Methodology: change training privacy while fixing the evaluation'),
            ('para','DP training clips each record’s gradient contribution and adds Gaussian noise. '
             'An accountant tracks a record-level (ε, δ) bound. I compare private and non-private '
             'MLPs on the same data, varying ε with δ=10⁻⁶ fixed. Smaller ε requires stronger '
             'privacy protection; the observed attack need not change monotonically. The implementation '
             'uses clipped, noisy Adam updates (DP-Adam), with Poisson sampling and RDP accounting.'),
            ('heading','Pipeline in the experiment'),
            ('pipeline',[
                'Freeze target partitions\nand public preprocessing',
                'Fit 16 mixed shadows;\nfreeze RF attack and cutoffs',
                'Train fresh non-private\nand seven private target MLPs',
                'Select weights using\npublic validation only',
                'Evaluate target utility\nand the same frozen MIA',
                'Report every ε with\naccuracy, AUC and low-FPR TPR']),
            ('small','Public 2014 preprocessing uses 10,000 rows; the other 173,941 rows supply shadows. '
             'The 16 non-private shadows are four each RF, gradient boosting, XGBoost and generic MLP, '
             'with own 40/20/40 splits and train-only preprocessing. The same 100,000 target train '
             'records and balanced 20,000 MIA candidates are used across all conditions.'),
            ('para','Targets use 512/256/128 hidden layers, dropout 0.1 and weight decay 0.0001, '
             'with a 30-epoch maximum and patience 5. Public-validation accuracy selects the returned '
             'weights, using loss to break ties. Noise is calibrated before training for the fixed '
             'maximum schedule; stopped-time ε is a diagnostic, not the claimed bound.'),
            ('heading','Saved results and a concrete interpretation'),
            ('table',(['Requested ε','Test accuracy %','MIA accuracy %','MIA AUC','TPR@1%FPR %'],dp_table,[.25,.19,.19,.15,.22])),
            ('para','For ε=1, the target correctly predicts income for 80.02% of test records; '
             'the transferred attack gets 49.705% membership accuracy and AUC 0.499. This describes '
             'the tested attack on a balanced sample. It does not mean ε=1 gives a 1% chance of '
             'leakage or that all possible attacks are ineffective.'),
            ('small','The same complete-feature attack is frozen across budgets; there is no missing-feature '
             'reconstruction in this DP notebook. ROC TPR@1%FPR uses target labels only to summarize '
             'the evaluation ROC. The auxiliary-selected cutoff is not recalibrated on target labels.')
        ]),
        ('5. Limitations and future work',
         'Interpreting the current evidence in relation to my proposal',[
            ('heading','Limitations'),
            ('para','1. Weak attack transfer. Auxiliary attack test AUC is 0.521 in Exp 3 and 0.513 in '
             'Exp 4. Shadows differ from the target in model family and overfitting. Near-chance DP '
             'attack results cannot establish resistance to stronger or DP-adapted attackers. '
             'Exp 3’s 62.625% accuracy also includes many false positives.'),
            ('para','2. Low-FPR calibration. The ROC TPR@1%FPR is an evaluation summary. At the auxiliary '
             'low-FPR cutoff, actual target FPR in the private runs is 7.18–11.68%, so those cutoff '
             'TPRs must not be described as detection at 1% target FPR.'),
            ('para','3. Data and masking assumptions. The 2014/2023 income distributions and category '
             'codes differ, and no inflation adjustment is used. Masks are artificial and related '
             'proxy fields remain visible. Year-qualified row-ID separation does not establish '
             'person-level independence. The local API simulation also omits real service restrictions.'),
            ('para','4. Reconstruction and walks. Forest donor sampling is an approximate distribution, '
             'not a calibrated posterior; six draws give a noisy SD estimate. SD may include finite-walk '
             'effects. Exp 3 retains a commuting-field mismatch: JWMNP/JWRIP use categorical shadow '
             'perturbations but numeric target perturbations. Exp 4 uses consistent rules.'),
            ('para','5. Repeatability and random-number settings. These are exploratory runs with one '
             'overall seed, not repeated whole experiments. The shadows have separate seeds, and '
             'private noise is independently sampled. Opacus secure_mode=False uses its research '
             'randomness setting; the setup is not hardened for deployment.'),
            ('para','6. Compute cost. DP training and repeated reconstruction queries are expensive on '
             'a laptop. Each completion can require up to 41 record predictions under the walk '
             'settings, multiplied by six draws and seven masks. This limits the number of repeat '
             'runs and alternative settings I can investigate.'),
            ('para','7. Privacy assumptions. For Experiment 4, target preprocessing data and target validation are assumed '
             'public and outside protected training records. The accountant guarantee is per-model '
             'and record-level, conditional on the fixed public data/partition. Releasing multiple '
             'models needs composition; the reported evaluation statistics are not DP-sanitized.'),
            ('heading','Future work'),
            ('para','Evaluate the combined uncertainty-aware MF-MIA against DP target models. '
             'For each privacy budget, hide candidate fields, generate conditional completions and '
             'compare single-draw, averaging, RF(mean) and RF(mean, SD) decisions against a matched '
             'non-private target. This would test whether DP changes the benefit of reconstruction '
             'variability. The combined experiment has not yet been run.'),
            ('small','Based on my proposal, “Uncertainty-Aware Membership Inference and Differential Privacy” '
             '(Q1, Q3, Q5, Q8), and existing saved results from the two named notebooks. '
             'Illustrative examples are explicitly labelled and are not reported as measured results. '
             'No notebook cells or experiments were rerun to prepare this document.')
        ])
    ]
    pdf_path=OUT/'methodology_pipeline_metrics.pdf'
    markdown=[]
    with PdfPages(pdf_path) as pdf:
        pdf.infodict().update(Title='Project explanation: membership inference and differential privacy',
                             Author=NAME,Subject='Methods, experiment pipelines, examples and saved results')
        for number,(title,subtitle,blocks) in enumerate(pages,1):
            page=ExplanationSheet(pdf,number,title,subtitle)
            markdown.extend([f'# {title}',subtitle,''])
            for kind,value in blocks:
                if kind=='heading':
                    page.heading(value);markdown.extend([f'## {value}',''])
                elif kind=='pipeline':
                    page.pipeline(value)
                    markdown.extend([' → '.join(label.replace('\n',' ') for label in value),''])
                elif kind=='branches':
                    page.branches(*value)
                    markdown.extend(['Target: '+' → '.join(value[0]).replace('\n',' '),
                                     'Auxiliary: '+' → '.join(value[1]).replace('\n',' '),
                                     value[2].replace('\n',' '),''])
                elif kind=='table':
                    page.table(*value)
                    headers,rows,_=value
                    markdown.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
                    markdown.extend('| '+' | '.join(map(str,row))+' |' for row in rows)
                    markdown.append('')
                else:
                    page.para(value,size=10.2 if kind=='small' else 11.2)
                    markdown.extend([value,''])
            page.finish()
    assert len(PdfReader(pdf_path).pages)==6
    doc='\n'.join(markdown)
    assert 'Questions for feedback' not in doc and 'remaining scope' not in doc.lower()
    (OUT/'project_explanation.md').write_text(doc,encoding='utf-8')
    sources=[PROPOSAL,ROOT/'exp_3_expanded_missing_feature_baseline.ipynb',
             ROOT/'exp_4_differential_privacy_accuracy.ipynb',E3/'summary.json',
             E3/'missing_feature_pfv/comparison.csv',E3/'uncertainty_mf_mia/comparison.csv',
             E3/'uncertainty_mf_mia/protocol.json',E4/'dp_target_and_mia_metrics.csv',E4/'protocol.json']
    manifest=dict(style='six-page student project explanation',pages=6,notebooks_executed=False,
                  prepared_date='2026-11-18',examples='Synthetic examples labelled; only tables labelled saved results are experimental observations',
                  sources=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources],
                  pdf_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest())
    data=json.dumps(manifest,indent=2)+'\n'
    (OUT/'source_manifest.json').write_text(data,encoding='utf-8')
    print(f'Created {pdf_path.relative_to(ROOT)} (6 pages). No experiments run.')


if __name__=='__main__':
    build_explanation()
