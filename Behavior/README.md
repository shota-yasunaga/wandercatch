# Behavior
Functions for behavioral analysis 
Information about the data index is at the bottom

## Scripts
- behavior_overview

  Plot what kind of events there were during 20 seconds before the probes. The trials go from left to right, top to bottom.
  (The figure generated contains 64 probes and it generates this for each participant)
  - Legend
    - The plot Background 
        - White  ... On Task
        - Yellow ... Mind Wandering
        - Blue   ... Mind Blanking
        - Grey   ... Didn't remember
    - The events 
        - Green ... Trial End
        - Red   ... Questions
        - Blue  ... Trial Start
        - Black ... Others

- transition_analysis
    
    Plots how the state (On task, Mind Wandering etc) transitioned.
- getLabelSequence.m

  Helper function for the transition_analysis. It returns the labels of probes
- behavior_analysis_ratio

    Plots information related how much of probes were "on task", "mind wandering", etc. 

- plot_conds_num_available
  
  plot number of participants available based on threshold for number of conditions for each conditions. 
  It plots 
  1. Number of available participants for cond vs cond (both conds have to have above threshold epochs available)
  2. Numbr of available participants for each cond


## Behavioral Data

- all_task_responses
 1. num_blocks
 2. num_trials
 3. doesn't_matter
 4. block_type ... 1: verbal task 2: spatial
 5. direction... 1 is right (forward),2 is lest(backward)
 6-9 ... right answer

 10-12... keyboard num (response)

 14-17... corrected response

 18-21... correctness-> NaN means there was probe
    --> no response time


- all_task_times
 1. Block Num
 2. Trial Num
 3. ??
 4. type (spatial/word)
 5. all 1s
 6. duplicate of 4
 7. ??
 8. ??
 9. Onset Trial
 10. Fixation
 11. Onset Stimulus presentation
 12. end of trial
 13. Onset que (arrow)
 14. Second arrow
 15. Num response
 - \*verbal  --> question onset is for each word
 16. Question for the first arrow
 17. Question for the second arrow
 18. Response for the first arrow
 19. Response for the second arrow
- \*spatial --> question onset is only one timing
 16. NaN
 17. NaN
 18. Answer for the first arrow
 19. Answer for the second arrow

- all_probe_response
 There are 7 questions for each probe
 1. Probe Num
 2. Block Num
 3. Trials Num (within a block)
 4. Probe Num (duplicate of 1)
 5. Question Num 
 6. Row keyboard Number
 7. Onset of Question
 8. Onset of the reaction
 9. What the answer was (for the question)
 
 For 2nd question (Labels)
 (1) Off-Task (2) Blank (3) Don''t Remember (4)


- all_probe_times
 1. Probe Num
 2. Block Num
 3. In which trial, this probe happened (WITHIN BLOCKS)
 4. Probe Num within the block
 5. Onset Probe
 6. Onset Response