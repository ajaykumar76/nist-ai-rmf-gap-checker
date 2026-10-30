"""
Build the synthetic evaluation data for the NIST AI RMF gap checker.

Writes:
  data/subcategories.json   24 PARAPHRASED AI RMF subcategories
  data/docs.json            12 synthetic governance documents with gold labels

Gold labels are assigned BY CONSTRUCTION: for each document and subcategory we decide the
status first (addressed / partial / absent) and then insert a matching sentence (or none).
That makes labels certain, but it also makes the task easier and cleaner than real policies.
The README says so. Re-running this script with the same seed reproduces the same files.

Run from the project root:  python src/00_build_dataset.py
"""
import json
import random
from pathlib import Path

SEED = 14

# (id, function, title, paraphrased summary, [3 addressed variants], [2 partial variants])
S = [
 ("GV-1.1", "GOVERN", "Legal and regulatory requirements",
  "Legal and regulatory requirements that apply to AI are identified, understood, managed and documented.",
  ["The Legal & Compliance team maintains a register of laws and regulations that apply to our AI use, including the EU AI Act and GDPR, and reviews it every quarter.",
   "Before any AI project starts, counsel confirms which statutory duties apply and records the answer in the project file.",
   "Applicable legal obligations for AI are tracked in a compliance matrix owned by the General Counsel and updated whenever the law changes."],
  ["We aim to stay aware of new AI regulation as it develops.",
   "Legal may be consulted on AI projects where it seems relevant."]),
 ("GV-1.6", "GOVERN", "AI system inventory",
  "Mechanisms exist to keep an inventory of AI systems, prioritised by risk.",
  ["Every AI system in use is entered in the central AI inventory, which records its owner, purpose and risk rating.",
   "IT keeps a living catalogue of all models and AI-enabled tools, and unregistered systems may not be deployed.",
   "A risk-ranked list of AI applications is maintained by the governance office and audited twice a year."],
  ["We have a spreadsheet of some of the AI tools teams told us about.",
   "An inventory of AI systems is planned for next year."]),
 ("GV-2.1", "GOVERN", "Roles and responsibilities",
  "Roles, responsibilities and lines of communication for AI risk are documented and clear to staff.",
  ["The Chief Data Officer is accountable for AI risk; model owners, reviewers and approvers are named in a RACI chart published on the intranet.",
   "Each AI system has a named business owner and a named technical owner, and escalation paths are written in the governance handbook.",
   "Responsibilities for AI oversight are assigned in writing, from the board risk committee down to individual model stewards."],
  ["Teams are generally expected to take responsibility for the AI they use.",
   "Roles for AI governance will be clarified in due course."]),
 ("GV-2.2", "GOVERN", "Training",
  "Staff and partners receive AI risk management training suited to their duties.",
  ["All staff who build or approve AI complete the annual AI risk training, and completion is tracked by HR.",
   "New hires in data and product roles take a mandatory AI ethics and risk course within 60 days, and contractors take it too.",
   "Role-based AI risk training is delivered each year, with advanced modules for model developers and reviewers."],
  ["Optional lunch-and-learn sessions on AI are sometimes offered.",
   "We encourage employees to learn about responsible AI on their own."]),
 ("GV-3.2", "GOVERN", "Human oversight roles",
  "Policies define the roles of humans in overseeing AI systems and in human-AI configurations.",
  ["Policy requires a named human reviewer to approve or override outputs of any AI system that affects customers, and the reviewer role is defined in the oversight standard.",
   "The oversight standard states which decisions need human approval, which need human review afterwards, and who performs each.",
   "For high-impact use cases, our policy defines when a person must be in the loop, on the loop, or notified."],
  ["People are expected to use their judgement when relying on AI output.",
   "We believe humans should stay involved in important AI decisions."]),
 ("GV-4.2", "GOVERN", "Impact documentation",
  "Teams document the risks and potential impacts of the AI they design, develop or deploy.",
  ["Project teams complete an AI impact assessment for each new system, recording potential harms to individuals and society, and file it with the governance office.",
   "A documented impact statement is required before deployment and is shared with affected business units.",
   "Teams must write up foreseeable risks and impacts of every AI system in a standard template that is reviewed by the risk function."],
  ["Teams are encouraged to think about possible impacts of their AI.",
   "Impact considerations are sometimes discussed in project meetings."]),
 ("GV-6.1", "GOVERN", "Third-party risk policy",
  "Policies address risks from third-party AI software, data and intellectual property.",
  ["Vendors supplying AI components must pass a security and AI-risk assessment, and contracts include clauses on data use, intellectual property and audit rights.",
   "Procurement applies a third-party AI risk checklist covering licensing, training data provenance and liability before any purchase.",
   "Our supplier policy requires disclosure of models, data sources and subcontractors for any AI product, and the clauses are reviewed by legal."],
  ["We prefer to work with reputable vendors.",
   "Third-party AI risk is something we intend to address in the supplier policy."]),
 ("MP-1.1", "MAP", "Intended purpose and context",
  "The intended purpose, context of use, users and applicable laws for each AI system are documented.",
  ["Each AI system has a charter that states its intended purpose, the users and context of use, and the laws that apply to it.",
   "Before development, the product owner documents the use case, target users, deployment setting and out-of-scope uses in a one-page brief.",
   "A system description is kept for every model, covering what it is for, who uses it, where it runs and what it must not be used for."],
  ["Teams generally know what their AI tools are meant to do.",
   "Project briefs sometimes mention the goal of the AI feature."]),
 ("MP-2.2", "MAP", "Knowledge limits",
  "Known limits of the system and how humans should use and oversee its output are documented.",
  ["Model cards list known limitations, failure modes and unsupported inputs, and state how staff should interpret and verify outputs.",
   "Documentation for each system describes where it performs poorly and tells operators when they must check results manually.",
   "User guides explain the system's limits and give instructions for human review of its outputs."],
  ["We tell users that the AI may sometimes be wrong.",
   "Limitations will be documented after launch."]),
 ("MP-3.5", "MAP", "Human oversight processes",
  "Processes for human oversight of the AI system are defined and assigned.",
  ["Each deployed system has a defined oversight procedure naming who monitors outputs, how often, and what they do when something looks wrong.",
   "Oversight workflows are written for every high-risk system, with assigned reviewers and sign-off steps.",
   "Operations runbooks describe how a human supervises the system day to day and who is on call."],
  ["Supervisors keep an eye on how the AI is working.",
   "Formal oversight procedures are under discussion."]),
 ("MP-4.1", "MAP", "Third-party components",
  "Legal and other risks of third-party components in the AI system are mapped.",
  ["For every external model, library or dataset we use, a component review records its licence, provenance and legal and security risks.",
   "The architecture review lists third-party AI components and assesses legal, copyright and supply-chain risk for each.",
   "A software bill of materials covers AI components, and legal reviews licences and data terms before use."],
  ["We try to use well-known open source models.",
   "Component risks are looked at when someone raises a concern."]),
 ("MP-5.1", "MAP", "Impact likelihood and magnitude",
  "The likelihood and magnitude of each identified positive and negative impact are assessed.",
  ["Risk assessments rate the likelihood and severity of each identified benefit and harm on a 5-point scale and are reviewed by the risk committee.",
   "For every AI use case we estimate how probable and how large the potential impacts are, and record the scores.",
   "The impact assessment includes a likelihood-and-magnitude matrix covering harms and benefits for all affected groups."],
  ["We consider whether bad outcomes seem likely.",
   "Impact scoring is expected to be introduced later."]),
 ("MS-1.1", "MEASURE", "Metrics selected",
  "Metrics and methods for measuring AI risks are chosen, and any risks left unmeasured are documented.",
  ["We track defined metrics for accuracy, robustness and harm for each system, and the risks we cannot yet measure are listed in the risk register.",
   "A measurement plan names the metrics, test methods and thresholds for each AI risk, with gaps recorded.",
   "Evaluation criteria are chosen per use case and approved by the model risk team, including a list of risks that are not currently measured."],
  ["We look at accuracy figures when they are available.",
   "Metrics will be chosen once the system is more mature."]),
 ("MS-1.3", "MEASURE", "Independent assessment",
  "Assessors independent of the development team take part in evaluating the system.",
  ["Model validation is carried out by a team independent of the developers, and its findings go to the risk committee.",
   "An external auditor reviews our high-risk systems every year.",
   "Independent reviewers who did not build the system test it before release and sign off separately."],
  ["Colleagues sometimes look over each other's work.",
   "We may ask an outside expert to review important systems in future."]),
 ("MS-2.4", "MEASURE", "Production monitoring",
  "System behaviour is monitored while in production.",
  ["Production models are monitored with dashboards for drift, error rates and unusual outputs, and alerts go to the on-call owner.",
   "Live system behaviour is tracked continuously against baseline performance, with thresholds that trigger investigation.",
   "Automated monitoring compares deployed model outputs with expected behaviour and logs deviations daily."],
  ["Developers check on the system from time to time.",
   "We plan to introduce monitoring after launch."]),
 ("MS-2.5", "MEASURE", "Validity and reliability",
  "The system is shown to be valid and reliable, with limits of generalisation documented.",
  ["Before release, each model is validated on held-out data and the results, including how far they generalise, are documented.",
   "Reliability testing under varied conditions is run and recorded for every system, with the limits of generalisation stated.",
   "A validation report shows the model performs consistently on representative data and notes where results may not transfer."],
  ["Models are tested before they go live.",
   "We assume the model works as the vendor describes."]),
 ("MS-2.7", "MEASURE", "Security and resilience",
  "Security and resilience of the system are evaluated and documented.",
  ["Penetration tests and adversarial testing are performed on every AI system, and security findings are tracked to closure.",
   "Security reviews cover model theft, data poisoning and prompt injection, and resilience tests are documented.",
   "The security team assesses each AI system for attacks and failure resilience before go-live and annually after."],
  ["AI systems fall under our general IT security policy.",
   "We will look at AI-specific attacks later."]),
 ("MS-2.10", "MEASURE", "Privacy risk",
  "Privacy risk of the system is examined and documented.",
  ["A privacy impact assessment is completed for every AI system that processes personal data, reviewed by the data protection officer.",
   "Privacy risks, including re-identification and data leakage, are examined and recorded before deployment.",
   "The DPO signs off on data protection assessments for AI projects, covering training data and outputs."],
  ["We comply with GDPR in general.",
   "Privacy is considered where personal data is obviously involved."]),
 ("MS-2.11", "MEASURE", "Fairness and bias",
  "Fairness and bias are evaluated and results documented.",
  ["Each model is tested for performance differences across demographic groups and the results and mitigations are documented.",
   "Bias and fairness evaluations are run before release and every six months, using defined fairness metrics.",
   "A fairness report is a required release artifact for any system affecting people."],
  ["We believe our systems treat people fairly.",
   "Fairness testing is on the roadmap."]),
 ("MS-3.3", "MEASURE", "End-user feedback",
  "Channels let end users and affected people report problems and appeal outcomes.",
  ["Customers and staff can report problems with AI decisions through a dedicated form and can appeal outcomes to a human reviewer, and reports feed our evaluation metrics.",
   "An appeals process lets affected people challenge AI-driven decisions, and complaint data is reviewed monthly.",
   "Users can flag incorrect or harmful outputs in the product, and flagged cases are triaged and analysed by the quality team."],
  ["People can contact customer service with any concerns.",
   "We welcome feedback from users."]),
 ("MG-1.4", "MANAGE", "Residual risk",
  "Risks that remain after treatment are documented and communicated to those affected.",
  ["After mitigation, the remaining risk for each system is recorded in the risk register and communicated to the system owner and affected stakeholders.",
   "Residual risks are documented in the release notes and accepted in writing by the business owner.",
   "The risk treatment plan lists risks that remain after controls and who has accepted them."],
  ["Some risk will always remain, and we are aware of that.",
   "Remaining risks are discussed informally."]),
 ("MG-2.4", "MANAGE", "Disengagement",
  "Mechanisms exist to supersede, disengage or deactivate a system that behaves outside its intended use.",
  ["Every AI system has a documented kill switch and rollback procedure, and an owner is authorised to disable it if it behaves outside its intended use.",
   "Runbooks describe how to deactivate or replace a model quickly, and the procedure is tested each year.",
   "If a system drifts outside its approved scope, the owner can suspend it immediately under the deactivation policy."],
  ["We could turn the system off if we had to.",
   "A shutdown procedure is being considered."]),
 ("MG-4.1", "MANAGE", "Post-deployment monitoring plan",
  "Post-deployment monitoring plans exist, covering user input, appeal and override, change management and decommissioning.",
  ["A post-deployment monitoring plan is required for each system, covering user feedback capture, appeal and override, change management and decommissioning.",
   "Each release includes a plan for ongoing monitoring, incident response, updates and eventual retirement of the model.",
   "Post-launch plans define how input from users is collected and evaluated and how the system will be changed or retired."],
  ["We intend to keep an eye on systems after launch.",
   "Teams handle updates as they see fit."]),
 ("MG-4.3", "MANAGE", "Incident communication",
  "Incidents and errors are communicated to relevant parties and affected communities, and response and recovery are tracked.",
  ["AI incidents are logged and reported to the governance committee within 72 hours, and affected users are notified according to the incident response policy.",
   "An incident register tracks AI errors, responses and recovery, with disclosure to affected parties.",
   "Significant AI failures are communicated to the relevant teams and customers, and the response is tracked until closure."],
  ["Serious problems would be escalated to management.",
   "We have a general process for handling incidents."]),
]

# Sentences that are about the organisation but are NOT evidence for any subcategory.
DISTRACTORS = [
 "The company was founded in 2009 and now operates in eleven countries.",
 "Our customer support team is available around the clock in six languages.",
 "We use machine learning to improve search ranking and product recommendations.",
 "The data science team is based in two offices and works in two-week sprints.",
 "Cloud infrastructure is provided under a multi-year contract with a major provider.",
 "The annual report is published each spring and covers financial performance.",
 "We are proud of our culture of innovation and customer focus.",
 "Our sustainability programme aims to reduce office energy use by 20 percent.",
 "Product launches follow a quarterly release calendar.",
 "The marketing team uses AI tools to draft campaign ideas.",
 "We value responsible and trustworthy AI as part of our mission.",
 "Our values are integrity, curiosity and respect for people.",
 "Technology is central to how we serve our customers.",
 "The board meets four times a year.",
 "We are committed to ethical business practices across the organisation.",
 "Employees can use the internal wiki to find team documentation.",
]

ORGS = ["Northwind Retail", "Alder Health", "Brightpath Bank", "Cobalt Logistics", "Delta Insure",
        "Evergreen Energy", "Fjord Telecom", "Granite Media", "Harbor Foods", "Iris Education",
        "Juniper Travel", "Kestrel Motors"]
MATURITY = ["low", "mid", "high", "mid", "low", "high", "mid", "low", "high", "mid", "low", "high"]
PROFILE = {"low": (0.15, 0.30), "mid": (0.40, 0.30), "high": (0.70, 0.20)}  # (P addressed, P partial)
SPLIT = ["dev"] * 8 + ["test"] * 4
SECTION_TEXT = {"GOVERN": "Governance", "MAP": "Mapping and context", "MEASURE": "Measurement and testing",
                "MANAGE": "Management and response"}


def build():
    rng = random.Random(SEED)
    docs = []
    # shuffle which orgs are test so maturity is mixed across splits
    order = list(range(12))
    rng.shuffle(order)
    for k, i in enumerate(order):
        split = SPLIT[k]
        org, mat = ORGS[i], MATURITY[i]
        p_add, p_par = PROFILE[mat]
        gold, per_func = {}, {f: [] for f in SECTION_TEXT}
        for sid, func, title, summ, addv, parv in S:
            r = rng.random()
            # Held-out wording: test documents use sentence variants that dev documents never use,
            # so thresholds and prompts tuned on dev have not seen the test sentences.
            add_pool = addv[:2] if split == "dev" else addv[2:]
            par_pool = parv[:1] if split == "dev" else parv[1:]
            if r < p_add:
                status = "addressed"
                per_func[func].append(rng.choice(add_pool))
            elif r < p_add + p_par:
                status = "partial"
                per_func[func].append(rng.choice(par_pool))
            else:
                status = "absent"
            gold[sid] = status
        test_idx = {i for i in range(len(DISTRACTORS)) if i % 3 == 1}
        pool = [d for i, d in enumerate(DISTRACTORS) if (i in test_idx) == (split == "test")]
        distract = rng.sample(pool, min(len(pool), rng.randint(4, 5) if split == "test" else rng.randint(7, 10)))
        for d in distract:
            per_func[rng.choice(list(SECTION_TEXT))].append(d)
        sentences, parts = [], [f"{org}: AI Governance Overview"]
        for f, head in SECTION_TEXT.items():
            rng.shuffle(per_func[f])
            sentences += per_func[f]
            parts.append(f"\n{head}\n" + " ".join(per_func[f]))
        docs.append({"id": f"D{k+1:02d}", "org": org, "maturity": mat, "split": split,
                     "text": "\n".join(parts), "sentences": sentences, "gold": gold})
    docs.sort(key=lambda d: d["id"])
    return docs


def main():
    out = Path("data")
    out.mkdir(exist_ok=True)
    subs = [{"id": sid, "function": f, "title": t, "summary": s} for sid, f, t, s, _, _ in S]
    note = ("PARAPHRASED summaries of NIST AI RMF 1.0 subcategories (a 24-item subset), not the official "
            "text. Verify wording at the NIST AI Resource Center. Not an official NIST product.")
    json.dump({"note": note, "subcategories": subs}, open(out / "subcategories.json", "w", encoding="utf-8"), indent=1)
    docs = build()
    json.dump({"note": "Synthetic documents; gold labels assigned by construction. Test documents use held-out sentence variants and distractors.", "docs": docs},
              open(out / "docs.json", "w", encoding="utf-8"), indent=1)
    from collections import Counter
    for split in ("dev", "test"):
        c = Counter(v for d in docs if d["split"] == split for v in d["gold"].values())
        n = sum(1 for d in docs if d["split"] == split)
        print(f"{split}: {n} docs, {sum(c.values())} labels, {dict(c)}")


if __name__ == "__main__":
    main()
