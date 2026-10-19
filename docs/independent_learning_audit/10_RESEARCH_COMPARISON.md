# 10. Современные методы: первичные источники, совместимость и неопределённость

Источник не равен нашему эксперименту. **VERIFIED** здесь означает существование/содержание указанной статьи или официального кода; применимость к нашему M4/ROSMASTER — обычно **HYPOTHESIS/UNKNOWN**. Новые модели и зависимости не устанавливались. Сравнительного обучения альтернатив не проводилось.

Даты относятся к указанной статье/релизу, не к вымышленному installed version. Если единого canonical paper/repo у семейства нет, это явно отмечено. Непроверенная аппаратная совместимость не заменяется догадкой по числу параметров. Данные для таблицы также в [research_sources.csv](research_sources.csv).

Важные аппаратные ограничения: Mac MPS не CUDA; массовая GPU-симуляция ManiSkill наMac не поддержана официальной таблицей; стабильнаяIsaacSim-ветка — Windows/Linux/NVIDIA, Newtonbeta рассматривать отдельно; JAX «работаетнаMac» не означает измереннуюAppleGPUскорость. Jetson inference и desktop training — разные задачи. Ресурсы из статей нельзя выдавать за доступные пользователю.

## M01. Behaviour Cloning

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2108.03298), [официальный код / project](https://github.com/ARISE-Initiative/robomimic); family;2021 benchmark exemplar.

**Принцип:** Supervised imitation of expert actions; a small feedforward/recurrent baseline separates pipeline defects from chunk-policy complexity.

**Данные:** Expert observation/action pairs; failures require corrections or a separate objective.

**CPU/GPU / Mac:** Small PyTorch baseline may run CPU/MPS; task-specific measurement required.

**MuJoCo / Jetson:** MuJoCo data adapter needed; policy inference can be portable.

**Ограничение и применение:** A cheap control baseline; not automatically robust to compounding errors.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M02. ACT

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2304.13705), [официальный код / project](https://github.com/tonyzhaozh/act); 2023-04.

**Принцип:** Predict action chunks from observations; original method includes latent modelling and temporal aggregation, disabled in this project.

**Данные:** Aligned demonstrations; same action semantics at train and execution.

**CPU/GPU / Mac:** Our reduced LeRobot variant actually trained on MPS; original full recipe is different.

**MuJoCo / Jetson:** Original simulated tasks use MuJoCo; Jetson current model still needs timing/contract tests.

**Ограничение и применение:** Keep as measured baseline; do not infer whole-room skill from local success.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M03. Diffusion Policy

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2303.04137), [официальный код / project](https://github.com/real-stanford/diffusion_policy); 2023-03.

**Принцип:** Conditional action-distribution denoising with receding-horizon execution can represent multiple action modes.

**Данные:** Demonstrations, compatible image/action encoders.

**CPU/GPU / Mac:** Official development Mac environment is not a benchmark of M4 training or Jetson latency.

**MuJoCo / Jetson:** MuJoCo/robomimic data possible through adapters; export and denoising latency must be measured.

**Ограничение и применение:** Relevant comparator for multimodality, not a fix for wrong targets/clock/units.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M04. DAgger / aggregation

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/1011.0686), [официальный код / project](https://github.com/HumanCompatibleAI/imitation); 2010 preprint / 2011.

**Принцип:** Query an expert on states visited by the learner and aggregate supervision.

**Данные:** Expert capable of correcting visited states; student predictions are not automatically expert targets.

**CPU/GPU / Mac:** Algorithm overhead small; expert/physics usually dominates. Mac simulator adapter needed.

**MuJoCo / Jetson:** Usable with our MuJoCo collector; final student inference need not carry expert.

**Ограничение и применение:** Our bounded prefix/correction is related, not a claim of reproducing all DAgger guarantees.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M05. Offline RL

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2110.06169), [официальный код / project](https://github.com/ikostrikov/implicit_q_learning); family; IQL2021 exemplar.

**Принцип:** Learn return-aware policy from a fixed transition dataset without new online action.

**Данные:** Rewards, next observation, terminal/truncation and sufficient support; current imitation rows need an audited MDP conversion.

**CPU/GPU / Mac:** State baselines modest; visual encoder/replay can dominate; MPS repository support unverified.

**MuJoCo / Jetson:** MuJoCo transitions possible; learned actor might deploy on Jetson after conversion.

**Ограничение и применение:** Failures can be used, but missing state/reward semantics cannot be cured by an offline-RL label.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M06. Online RL

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/1801.01290), [официальный код / project](https://github.com/rail-berkeley/softlearning); family; SAC2018 exemplar.

**Принцип:** Collect experience and optimize reward through interaction rather than only imitation.

**Данные:** Resettable environment, trustworthy reward, safety and observation contract.

**CPU/GPU / Mac:** CPU rollout possible; visual/parallel simulation and RL update cost must be budgeted.

**MuJoCo / Jetson:** Custom MuJoCo environment needed; actor deployable in principle, no current ROSMASTER test.

**Ограничение и применение:** Autonomous simulation is plausible; autonomous safe real exploration is not granted.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M07. SAC

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/1801.01290), [официальный код / project](https://github.com/rail-berkeley/softlearning); 2018-01.

**Принцип:** Off-policy entropy-regularized actor/critic with replay.

**Данные:** Transitions/reward; demonstrations optional but helpful for sparse grasp.

**CPU/GPU / Mac:** Current project has no full SAC run; MPS/CPU feasibility is a hypothesis for chosen implementation.

**MuJoCo / Jetson:** MuJoCo natural fit; Jetson actor may be small, visual latency must be tested.

**Ограничение и применение:** Candidate for short goal-conditioned skills after valid MDP/reward/reset.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M08. PPO

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/1707.06347), [официальный код / project](https://github.com/openai/baselines); 2017-07.

**Принцип:** Clipped on-policy policy-gradient updates on newly sampled rollouts.

**Данные:** Many safe simulated transitions and useful reward/curriculum.

**CPU/GPU / Mac:** CPU MuJoCo works in principle; no claim of massive Apple GPU flex simulation.

**MuJoCo / Jetson:** Custom environment and actor adapters; deployment policy often smaller than training infrastructure.

**Ограничение и применение:** A fair pilot is possible; sparse deformable pick is not solved merely by choosing PPO.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M09. HIL-SERL

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2410.21845), [официальный код / project](https://github.com/rail-berkeley/hil-serl); 2024-10.

**Принцип:** Sample-efficient off-policy robotics RL with demonstrations, interventions and reward infrastructure.

**Данные:** Human interventions/reset/reward provisioning are material requirements, unlike a fully autonomous promise.

**CPU/GPU / Mac:** Official stack is not measured on our M4/servo arm. Porting robot interface is required.

**MuJoCo / Jetson:** Simulation adaptation possible; real published platform results do not transfer automatically to ROSMASTER.

**Ограничение и применение:** Useful design lessons; our tiny linear actor/critic is not HIL-SERL.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M10. RLPD

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2302.02948), [официальный код / project](https://github.com/ikostrikov/rlpd); 2023-02.

**Принцип:** Efficient online RL that mixes offline prior data and online replay.

**Данные:** Prior transitions plus new online environment and rewards.

**CPU/GPU / Mac:** Reference JAX stack includes MuJoCo/D4RL tasks; Apple GPU support not verified here.

**MuJoCo / Jetson:** MuJoCo adapter plausible; Jetson inference requires compatible actor export.

**Ограничение и применение:** Relevant after transition conversion; cannot reuse raw failed videos as correct replay blindly.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M11. IQL

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2110.06169), [официальный код / project](https://github.com/ikostrikov/implicit_q_learning); 2021-10.

**Принцип:** Offline value learning and advantage-weighted policy extraction without arbitrary out-of-dataset action evaluation.

**Данные:** Transition/reward dataset; action support still matters.

**CPU/GPU / Mac:** Official JAX/D4RL recipe not validated on M4; CPU and alternative implementations differ.

**MuJoCo / Jetson:** MuJoCo precedents; our cloth and action contract are new tasks.

**Ограничение и применение:** Comparator for audited fixed dataset, not a guarantee of recovery outside support.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M12. TD3+BC

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2106.06860), [официальный код / project](https://github.com/sfujim/TD3_BC); 2021-06.

**Принцип:** Simple offline actor/critic plus behaviour-cloning regularization.

**Данные:** Continuous transitions/reward; mostly state-control reference benchmarks.

**CPU/GPU / Mac:** Reference PyTorch recipe needs adaptation/testing for MPS and visual observations.

**MuJoCo / Jetson:** MuJoCo benchmarks; potentially compact Jetson actor, untested on this robot.

**Ограничение и применение:** Strong inexpensive state baseline; comparison must match observation availability.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M13. robomimic

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2108.03298), [официальный код / project](https://github.com/ARISE-Initiative/robomimic); 2021-08; docs0.5.

**Принцип:** Framework and benchmarks for evaluating imitation from human demonstrations, not one universal policy.

**Данные:** Structured demonstrations and controlled splits.

**CPU/GPU / Mac:** PyTorch; memory depends on chosen policy. Mac compatibility is not our measured M4 benchmark.

**MuJoCo / Jetson:** Robosuite/MuJoCo integration; own embodiment/control adapter needed for ROSMASTER.

**Ограничение и применение:** Useful BC/BC-RNN baselines and data conventions; preserve independent task outcomes.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M14. MimicGen

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2310.17596), [официальный код / project](https://github.com/NVlabs/mimicgen); 2023-10.

**Принцип:** Transform segmented source demonstrations relative to objects to generate new trajectories.

**Данные:** Seed demos, subtask annotations/object transforms and valid execution.

**CPU/GPU / Mac:** CPU MuJoCo data generation possible; complex rendering/training may need GPU. No M4 measurement here.

**MuJoCo / Jetson:** Robosuite/MuJoCo; deformable sock manipulation and ROSMASTER controller need custom work.

**Ограничение и применение:** Promising data-generation design; not zero-demonstration universal teacher.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M15. RoboCasa / RoboCasa365

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2406.02523), [официальный код / project](https://github.com/robocasa/robocasa); 2024; release2026-02.

**Принцип:** Task/scene/data/benchmark framework for diverse household manipulation; recent releases add structured subtasks.

**Данные:** Assets/tasks/demonstrations; existing embodiments differ.

**CPU/GPU / Mac:** MuJoCo-based environment can run on major platforms; heavy visual generation still needs measured resources.

**MuJoCo / Jetson:** Custom robot and controller port required; kitchen demonstrations do not validate our cloth task.

**Ограничение и применение:** Useful scene/protocol ideas, not drop-in coverage of ROSMASTER and floor socks.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M16. ManiSkill3

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2410.00425), [официальный код / project](https://github.com/mani-skill/ManiSkill); 2024 preprint / RSS2025.

**Принцип:** Parallelized simulation/rendering and robotics learning baselines with SAPIEN.

**Данные:** New embodiment/assets/reward/controller/task ports.

**CPU/GPU / Mac:** Official table: macOS CPU simulation/rendering, no Mac GPU simulation; Linux/NVIDIA for GPU sim.

**MuJoCo / Jetson:** Not MuJoCo physics; port may change cloth/contact behaviour. Jetson actor export remains separate.

**Ограничение и применение:** Scale candidate if resources justify port; not a switch for existing MuJoCo flex scenes.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M17. Isaac Lab

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2511.04831), [официальный код / project](https://github.com/isaac-sim/IsaacLab); 2025 paper; stable/beta docs checked2026-10-19.

**Принцип:** Robot-learning framework with physics/rendering integrations; stable Isaac Sim path and experimental Newton paths differ.

**Данные:** Converted robot/assets/tasks and validated reward/control.

**CPU/GPU / Mac:** Opened stable installation: Windows/Linux, NVIDIA/Isaac Sim,16GB+VRAM; not native M4 route. Beta support must be checked separately.

**MuJoCo / Jetson:** Not byte-equivalent MuJoCo flex port; training and Jetson policy deployment are different concerns.

**Ограничение и применение:** Potential future NVIDIA scaling, not proven benefit on user hardware today.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M18. LeRobot

**Источники / дата:** [статья или первичное описание](https://huggingface.co/docs/lerobot/index), [официальный код / project](https://github.com/huggingface/lerobot); framework; installed0.6.1.

**Принцип:** Dataset/processors/policies/training/tooling framework, not an independent learning algorithm.

**Данные:** Embodiment-specific datasets and calibrated action schema.

**CPU/GPU / Mac:** Our ACT/MPS works; that does not validate every backend or external package version.

**MuJoCo / Jetson:** Portable dataset/policy interfaces; no automatic validated ROSMASTER world model.

**Ограничение и применение:** Retain provenance/resume/processors; fix integration rather than assuming framework solves task.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M19. SmolVLA

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2506.01844), [официальный код / project](https://github.com/huggingface/lerobot); 2025-06.

**Принцип:** Compact pretrained language-conditioned visual action model,450M; supports fine-tuning.

**Данные:** Own robot data/action adaptation; community pretraining not ROSMASTER calibration.

**CPU/GPU / Mac:** Consumer hardware orientation; actual fine-tune memory/latency on our M4/Jetson untested.

**MuJoCo / Jetson:** MuJoCo data adapter possible; Jetson8GB fit/latency must be measured, not inferred from parameter count.

**Ограничение и применение:** Relevant multi-task candidate after working contract; existing adapter is not accepted policy.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M20. OpenVLA / OFT

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2406.09246), [официальный код / project](https://github.com/openvla/openvla); 2024-06; OFT2025-02.

**Принцип:** Pretrained7B visual-language-action policy; OFT separately improves action/fine-tuning execution.

**Данные:** Embodiment-specific demonstrations, normalization/action adapter.

**CPU/GPU / Mac:** Official original LoRA example:80GB A100, lower bound about27GB by reducing batch; not demonstrated on24GB unified Mac.

**MuJoCo / Jetson:** Simulation data possible;8GBJetson inference needs separate quantization/latency research.

**Ограничение и применение:** Not a justified immediate replacement; extra compute does not fix our invalid state distribution.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M21. VLM-guided task/experiment planning

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2204.01691), [официальный код / project](https://github.com/google-research/google-research/tree/master/saycan); SayCan2022; Google update2026-10-13.

**Принцип:** Language/vision model proposes goals or skills; grounding/feasibility/safety decides execution.

**Данные:** Capability library, observations, tool contracts and independently verified outcomes.

**CPU/GPU / Mac:** Remote or local VLM costs differ; cloud result does not imply on-device Jetson model.

**MuJoCo / Jetson:** Can propose MuJoCo scenes/diagnoses; generated code/scenes must pass geometry and test gates.

**Ограничение и применение:** Useful experiment proposal layer; never sole judge or unrestricted actuator owner.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M22. Hierarchical RL / options

**Источники / дата:** [статья или первичное описание](https://doi.org/10.1016/S0004-3702(99)00052-1), `UNKNOWN_CANONICAL_REPOSITORY`; 1999.

**Принцип:** Temporally extended options have initiation, closed-loop policy and termination; hierarchical learning is not only a fixed stage list.

**Данные:** Option/goal rewards, reliable terminations and learning signals.

**CPU/GPU / Mac:** Costs depend on skills and algorithm; no single hardware requirement or stock implementation.

**MuJoCo / Jetson:** MuJoCo feasible through environment integration; Jetson can compose accepted skills.

**Ограничение и применение:** Promising decomposition; our scripted stages are not proof of learned options.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M23. Goal-conditioned policies / HER

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/1707.01495), [официальный код / project](https://github.com/openai/baselines/tree/master/baselines/her); 2017-07 exemplar.

**Принцип:** Condition policy on goal and relabel achieved goals to reuse experience.

**Данные:** Observable achieved/desired goal, valid relabelable reward; hidden truth cannot become real actor input.

**CPU/GPU / Mac:** State-policy pilots modest; visual/cloth goals need explicit representation and tests.

**MuJoCo / Jetson:** Custom MuJoCo goals possible; deployment must estimate goal from real sensors.

**Ограничение и применение:** Could remove phase-time shortcut; reward relabelling validity for cloth requires proof.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M24. Visual servoing

**Источники / дата:** [статья или первичное описание](https://faculty.cc.gatech.edu/~seth/ResPages/pdfs/HutHagCor96.pdf), [официальный код / project](https://github.com/lagadic/visp); 1996 / 2006.

**Принцип:** Close control loop on measured visual features/image error rather than only command history.

**Данные:** Tracked features/geometry, calibrated camera/Jacobian and observable target.

**CPU/GPU / Mac:** Generally CPU viable; sensor latency/occlusion matters more than model size.

**MuJoCo / Jetson:** MuJoCo camera tests straightforward; real camera delay/backlash still must be measured.

**Ограничение и применение:** Strong baseline for alignment/visibility, not a universal deformable grasp solver.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M25. Task and Motion Planning / PDDLStream

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/1802.08705), [официальный код / project](https://github.com/caelan/pddlstream); 2018 preprint / ICAPS2020.

**Принцип:** Combine symbolic planning with black-box geometric samplers and feasibility checks.

**Данные:** Explicit domain/actions, samplers, collision geometry, observations.

**CPU/GPU / Mac:** Mostly CPU search; complexity depends on task/geometric branching.

**MuJoCo / Jetson:** MuJoCo not required; adapters can use its geometry; deploy planning with calibrated real state.

**Ограничение и применение:** Useful typed task compiler; does not automatically learn missing motor skill.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M26. Residual learning / residual RL

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/1812.03201), `UNKNOWN_OFFICIAL_REPOSITORY`; 2018-12.

**Принцип:** Learn bounded correction around an existing controller instead of replacing all control.

**Данные:** A genuinely useful baseline, residual limits and reward/demonstrations.

**CPU/GPU / Mac:** Small residual actor may be modest; learning stack/platform still needs a pilot.

**MuJoCo / Jetson:** MuJoCo feasible; baseline/learned residual interface must match real actuation.

**Ограничение и применение:** Candidate only if baseline has actual reliable success; not a disguise for parameter-only search.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M27. Domain randomization

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/1703.06907), `UNKNOWN_CANONICAL_REPOSITORY`; 2017-03.

**Принцип:** Vary rendering/dynamics to improve robustness over a specified distribution.

**Данные:** Plausible parameter ranges and evaluation outside training samples.

**CPU/GPU / Mac:** Scene/rollout generation dominates; no automatic Apple GPU acceleration.

**MuJoCo / Jetson:** Already bounded MuJoCo variation exists; real transfer unverified.

**Ограничение и применение:** Keep measured relevant variability; random colours do not cover unseen recovery states.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M28. System identification / SimOpt

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/1810.05687), [официальный код / project](https://sites.google.com/view/simopt); 2018-10.

**Принцип:** Adapt simulator randomization using real-world experience rather than arbitrary estimated parameters.

**Данные:** Comparable real/sim trajectories and meaningful discrepancy metric.

**CPU/GPU / Mac:** Optimization needs repeated sim evaluations; hardware costs depend on chosen model.

**MuJoCo / Jetson:** MuJoCo-compatible in principle; current instance lacks complete real dynamics measurements.

**Ограничение и применение:** Important before claiming sim-to-real; website is project resource, not verified code repository.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M29. Curriculum learning

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/1707.05300), `UNKNOWN_CANONICAL_REPOSITORY`; family; robotic automatic curricula2017.

**Принцип:** Choose progressively useful start states/goals/difficulties based on measured learning progress.

**Данные:** Success/failure diagnostics and viable reset generation.

**CPU/GPU / Mac:** Costs largely task rollout; our parameter-phase scheduler is a narrow example.

**MuJoCo / Jetson:** MuJoCo feasible; deployment does not require curriculum runtime.

**Ограничение и применение:** Choose failure-relevant states, not repeat two easy scenes or merely increase attempt cap.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M30. Automatic domain generation / ADR

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/1910.07113), [официальный код / project](https://github.com/openai/robogym); 2019-10.

**Принцип:** Expand or adapt randomization/task distribution through training feedback.

**Данные:** Parameterized simulator, valid task/reset/reward and adequate compute.

**CPU/GPU / Mac:** Large published scaling setups are not proof of feasibility on24GB M4.

**MuJoCo / Jetson:** MuJoCo-related environment resources; robotics domain port and cloth validation still needed.

**Ограничение и применение:** Useful generator principle; does not invent arbitrary valid new tasks from a sentence.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M31. World models / DreamerV3

**Источники / дата:** [статья или первичное описание](https://www.nature.com/articles/s41586-025-08744-2), [официальный код / project](https://github.com/danijar/dreamerv3); 2023 method / Nature2025.

**Принцип:** Learn latent dynamics/reward and train actor/critic using imagined trajectories.

**Данные:** Real/environment transitions and valid reward; model errors can compound in contact tasks.

**CPU/GPU / Mac:** Official code tested Linux/Mac; JAX Apple GPU speed and our task memory not measured.

**MuJoCo / Jetson:** Custom MuJoCo integration; Jetson actor plus observation encoder requires export testing.

**Ограничение и применение:** Worth a separate pilot only after observation/reward/reset contract; not a shortcut around physical uncertainty.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M32. Model-based RL / TD-MPC2

**Источники / дата:** [статья или первичное описание](https://arxiv.org/abs/2310.16828), [официальный код / project](https://github.com/nicklashansen/tdmpc2); 2023-10 / ICLR2024.

**Принцип:** Learn implicit latent dynamics and optimize short-horizon actions with value learning.

**Данные:** Transitions/reward; state and pixel variants need appropriate encoders.

**CPU/GPU / Mac:** Official experiments use NVIDIA/PyTorch ecosystem; M4 support not tested here.

**MuJoCo / Jetson:** MuJoCo tasks have precedents; online planning latency on Jetson is a separate constraint.

**Ограничение и применение:** Could test dynamics-aware residual/control; cloth/partial observability remain risks.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M33. openpi / pi0.5

**Источники / дата:** [статья или первичное описание](https://www.physicalintelligence.company/blog/pi05), [официальный код / project](https://github.com/Physical-Intelligence/openpi); pi0.5 release2025-09.

**Принцип:** Pretrained generalist action model with task-specific fine-tuning; action head/embodiment adaptation matters.

**Данные:** Robot data/normalization and mapping to our6servo+base action space.

**CPU/GPU / Mac:** Official single NVIDIA GPU estimates: inference>8GB,LoRA>22.5GB,full>70GB. Unified24GB Mac is not equivalent available VRAM.

**MuJoCo / Jetson:** MuJoCo data possible; standard Nano8GB route is not established. Remote inference is a different deployment design.

**Ограничение и применение:** Potential comparator with justified compute, not a currently installed ROSMASTER learner.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## M34. Isaac GR00T N1.7

**Источники / дата:** [статья или первичное описание](https://github.com/NVIDIA/Isaac-GR00T), [официальный код / project](https://github.com/NVIDIA/Isaac-GR00T); current repository checked2026-10-19.

**Принцип:** Generalist robot foundation model with embodiment/data adaptation workflows.

**Данные:** Own embodiment mapping, task data and compatible observation modalities.

**CPU/GPU / Mac:** NVIDIA-oriented stack; precise current Mac/Jetson support and minimum for this task UNKNOWN, not benchmarked.

**MuJoCo / Jetson:** Simulation adaptation is not accepted real transfer; ROSMASTER servo semantics are not stock supported proof.

**Ограничение и применение:** Include in open comparison, but not assume hardware or skill compatibility from model name.

**Наш статус:** кроме действующего уменьшенногоACT, сравнительная обученная/принятая реализация этого метода наROSMASTER отсутствует. Симуляционное обучение возможно только после определения корректного expert/reward/reset; необходимость реальных показов зависит от задачи, но реальные измерения и проверка переноса не отменяются.

## VLM как инженерный помощник

[Официальный Google Developers Blog13.10.2026](https://developers.googleblog.com/gemini-4-argon-for-robotics/) описывает VLM-driven робототехнические workflows. Это основание рассмотреть генерацию гипотез/экспериментов, но не передать модели неограниченное управление. Текущий аудит не отправлял приватные кадры внешнемуAPI и не проверял данные производителя наROSMASTER. Никакая такая новость не заменяет наш независимый evaluator.

## Что выбирать только экспериментом

Сначала один небольшой корректный датасет и одинаковый runtime: сравнить BC/BC-RNN, ACT и Diffusion Policy. Затем, если готовы MDP, reward и reset, провести ограниченный SAC/RLPD/PPO pilot. После измерения требований к языку и многозадачности рассматривать SmolVLA и другие VLA. TAMP и visual servoing могут уменьшить ошибки восприятия и планирования, но их вклад нужно измерять по полной миссии. Новая архитектура заранее не назначена победителем.
