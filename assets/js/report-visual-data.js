window.REPORT_VISUAL_DATA = {
  finalResults: {
    textMetrics: {
      labels: ["Exact-match", "Micro F1", "Macro F1"],
      datasets: [
        {
          label: "BERT",
          values: [0.9306304048126331, 0.8029761053083416, 0.6764643489662164],
        },
        {
          label: "LSTM",
          values: [0.9223586915653591, 0.7386150966936993, 0.5489344217823646],
        },
      ],
    },
    textLoss: {
      labels: [1, 2, 3, 4, 5],
      suggestedMax: 0.08,
      datasets: [
        {
          label: "BERT train loss",
          values: [0.06514481596685694, 0.030699367290163412, null, null, null],
        },
        {
          label: "BERT val loss",
          values: [0.03846429874065434, 0.0374828478464211, null, null, null],
        },
        {
          label: "LSTM train loss",
          values: [0.07576669362936178, 0.048532801606974804, 0.04080355510157638, 0.03441236961602623, 0.028841614143871572],
        },
        {
          label: "LSTM val loss",
          values: [0.0564214408993721, 0.05098114086687565, 0.0488917596526444, 0.04892577413469553, 0.051036074271425605],
        },
      ],
    },
    multimodalMetrics: {
      labels: ["Accuracy", "Macro F1"],
      datasets: [
        {
          label: "CLIP",
          values: [0.8806282722513089, 0.7965160351105383],
        },
        {
          label: "VisualBERT",
          values: [0.8115183246073299, 0.7290071095317586],
        },
      ],
    },
    multimodalLoss: {
      labels: [1, 2, 3],
      suggestedMax: 1.3,
      datasets: [
        {
          label: "CLIP train loss",
          values: [1.2033616679772574, 0.9302125922068248, 0.7633176113980835],
        },
        {
          label: "CLIP val loss",
          values: [1.1016062787157261, 0.9611351879421838, 0.9119733496753868],
        },
        {
          label: "VisualBERT train loss",
          values: [1.0126666838258815, 0.5618558598833856, 0.297288941765267],
        },
        {
          label: "VisualBERT val loss",
          values: [0.7239629891598391, 0.9622972936701694, 1.0505343315015194],
        },
      ],
    },
  },
  text: {
    metrics: {
      labels: ["Exact-match", "Micro F1", "Macro F1"],
      datasets: [
        {
          label: "BERT",
          values: [0.9306304048126331, 0.8029761053083416, 0.6764643489662164],
        },
        {
          label: "LSTM",
          values: [0.9223586915653591, 0.7386150966936993, 0.5489344217823646],
        },
      ],
    },
    perLabel: {
      labels: ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
      datasets: [
        {
          label: "BERT",
          values: [0.8478045559590623, 0.44755244755244755, 0.8421052631578947, 0.56, 0.7898952556993223, 0.5714285714285714],
        },
        {
          label: "LSTM",
          values: [0.7937995674116799, 0.37992831541218636, 0.7997535428219347, 0.32727272727272727, 0.7071380920613742, 0.2857142857142857],
        },
      ],
    },
    history: {
      labels: [1, 2, 3, 4, 5],
      datasets: [
        {
          label: "BERT val micro F1",
          values: [0.7730080296479308, 0.8008379888268157, null, null, null],
        },
        {
          label: "LSTM val micro F1",
          values: [0.7129461584996976, 0.7078052034689793, 0.7213691618682022, 0.7470507124253103, 0.7465940054495913],
        },
      ],
    },
    loss: {
      labels: [1, 2, 3, 4, 5],
      suggestedMax: 0.08,
      datasets: [
        {
          label: "BERT train loss",
          values: [0.06514481596685694, 0.030699367290163412, null, null, null],
        },
        {
          label: "BERT val loss",
          values: [0.03846429874065434, 0.0374828478464211, null, null, null],
        },
        {
          label: "LSTM train loss",
          values: [0.07576669362936178, 0.048532801606974804, 0.04080355510157638, 0.03441236961602623, 0.028841614143871572],
        },
        {
          label: "LSTM val loss",
          values: [0.0564214408993721, 0.05098114086687565, 0.0488917596526444, 0.04892577413469553, 0.051036074271425605],
        },
      ],
    },
  },
  multimodal: {
    metrics: {
      labels: ["Accuracy", "Macro F1"],
      datasets: [
        {
          label: "CLIP",
          values: [0.8806282722513089, 0.7965160351105383],
        },
        {
          label: "VisualBERT",
          values: [0.8115183246073299, 0.7290071095317586],
        },
      ],
    },
    perLabel: {
      labels: [
        "affected_individuals",
        "infrastructure_and_utility_damage",
        "rescue_volunteering_or_donation_effort",
        "other_relevant_information",
        "not_humanitarian",
      ],
      datasets: [
        {
          label: "CLIP",
          values: [0.5, 0.8765432098765432, 0.828125, 0.8774193548387097, 0.9004926108374385],
        },
        {
          label: "VisualBERT",
          values: [0.47619047619047616, 0.7560975609756098, 0.7857142857142857, 0.772823779193206, 0.8542094455852156],
        },
      ],
    },
    history: {
      labels: [1, 2, 3],
      datasets: [
        {
          label: "CLIP val macro F1",
          values: [0.6853765552491893, 0.7205963776426213, 0.7541132425607275],
        },
        {
          label: "VisualBERT val macro F1",
          values: [0.7136671048581228, 0.7211096075929481, 0.7485352166489834],
        },
      ],
    },
    loss: {
      labels: [1, 2, 3],
      suggestedMax: 1.3,
      datasets: [
        {
          label: "CLIP train loss",
          values: [1.2033616679772574, 0.9302125922068248, 0.7633176113980835],
        },
        {
          label: "CLIP val loss",
          values: [1.1016062787157261, 0.9611351879421838, 0.9119733496753868],
        },
        {
          label: "VisualBERT train loss",
          values: [1.0126666838258815, 0.5618558598833856, 0.297288941765267],
        },
        {
          label: "VisualBERT val loss",
          values: [0.7239629891598391, 0.9622972936701694, 1.0505343315015194],
        },
      ],
    },
  },
};
