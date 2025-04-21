## The problem at hand

Build a RL agent to optimize traffic light switching times, in a multi intersection context. The optimization is measured by checking the queue lengths, waiting time, throughput and other metrics as required.

### What makes this difficult?

There's a plethora of works on single intersection signal optimization. Most prominent of these is the open source code sample by AndreaVidali, available on Github, on which all later implementations are based. All of these use SUMO, a traffic simulation software for designing and simulating traffic patterns, with varying degrees of realism.

The issue is, that there are no real samples of RL applied in the multi intersection environment context. Various papers have been published, with different algorithms and environment configurations, but none of them are open source, so we cannot say that there is any significant work on this topic. I aim to address this by building a multi agent collaborative system of agents capable of achieving the best results in a global context.

However, this implementation is simplified, and will probably fail with real network layouts and traffic patterns by design. There are improvements which can be done to fix those but it would take a many months to implement successfully.

