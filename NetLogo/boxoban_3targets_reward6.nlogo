extensions [ qlearningextension csv string ]

globals [ matrix
  length-matrix
  target-count
  f-cnt ;; counts the number of boxes currently on the target
  f-cnt-old ;; last step count of the number of boxes on a target
  is-end-state? ;; either max steps, deadlock or successfully solved
  total-reward
  max-boxes-on-target ;;  maximum number of boxes simultaneously on a target this episode
  max-boxes-on-target-overall ;; maximum number of boxes simultaneously whole training
  reward-current-episode ;; current reward
  box-in-corner? ;; flag indicating if a box is stuck in a corner
  id-list ;; list of unique ids for labelling the floor tiles
  boxes-order ;; ordered list of all boxes used for state calculation
  times-finished ;; total times successfully solved
  episode-cnt
  accumulated-reward ;; accumulated reward over one epoch = 100 episodes
  times-finished-epoch ;; times finished current epoch
  steps ;; steps in current episode
  epoch-cnt ;; total epochs
  seed
  deadlock
  total-deadlock-cnt
  reward-last-epoch
  reward-last-episode
]

breed [ walls wall ]
breed [ workers worker ]
breed [ targets target ]
breed [ boxes box ]

patches-own [ is-empty? has-target? id ]

workers-own [ x-start y-start on-position current-state ]
boxes-own [ x-start y-start on-position last-position ]
targets-own [ on-position ]


to setup
  clear-all
  set seed new-seed
  random-seed seed
  set total-deadlock-cnt 0
  set reward-last-epoch 0
  set reward-last-episode 0
  set steps 0
  set accumulated-reward 0
  set total-reward 0
  set reward-current-episode 0
  set times-finished 0
  set episode-cnt 0
  set epoch-cnt 0
  set times-finished-epoch 0
  set id-list ( range 1 99 )
  read-text-file
  setup-world
  set max-boxes-on-target-overall 0
  ask workers [
    set x-start [ pxcor ] of patch-here
    set y-start [ pycor ] of patch-here
  ]
  ask boxes [
    set x-start [ pxcor ] of patch-here
    set y-start [ pycor ] of patch-here
  ]
  set boxes-order sort-on [ who ] boxes
  ask workers [
    set on-position [ id ] of patch-here
  ]
  ask boxes [ set on-position [ id ] of patch-here ] ;; to initialize before last-position is set in update-ids
  update-ids
  ask workers [
    set current-state calculate-state
    qlearningextension:state-def [ "current-state" ]
    (qlearningextension:actions [ move-up ] [ move-right ] [ move-down ]  [ move-left ] )
    qlearningextension:reward [ reward ]
    qlearningextension:end-episode [ is-end-state? ] reset-episode
    qlearningextension:action-selection "e-greedy" [ 1 0.9995 ]
    qlearningextension:learning-rate learning-rate
    qlearningextension:discount-factor disc-factor
  ]
  reset-timer
  reset-ticks
end


to read-text-file
  set matrix []
  set length-matrix 12
  let line-cnt 0
  file-open "level_0.txt"
  while [not file-at-end?] [
    let this-line file-read-line ; read next line
    let this-list []
    if ( line-cnt >= 1 ) and ( line-cnt < 11 ) [
      let i 1 ;; read line, first element is space
      while [ i < length this-line ] [ ;loop over line
        set this-list lput item i this-line this-list  ; get each character and put into list
        set i ( i + 1 )
      ]
      ;; print this-list
      set matrix lput this-list matrix ; add list to matrix (list of lists)
      set length-matrix length matrix
    ]
    set line-cnt line-cnt + 1
  ]
  file-close-all
end

;; used visualize the world
to setup-world
  set is-end-state? false
  set target-count 0
  set max-boxes-on-target 0
  let y-size length matrix
  let x-size length item 0 matrix
  set-patch-size 30
  resize-world 0 ( x-size - 1 ) 0 ( y-size - 1 )
  ask patches [ set is-empty? true set has-target? false ]
  let placed-targets-cnt 0
  let placed-boxes-cnt 0
  foreach range y-size [ y -> ;; loop over each line
    let this-line item ( y-size - y - 1 ) matrix
    foreach range x-size [ x -> ; loop over columns
      let this-item item x this-line
      (ifelse
        this-item = "#" [
          ask patch x y [
            sprout-walls 1
            set is-empty? false
            set pcolor grey
          ]
        ]
        this-item = "." [
          if placed-targets-cnt < max-targets [
            ask patch x y [
              sprout-targets 1
              set has-target? true
              set target-count ( target-count + 1 )
              set placed-targets-cnt ( placed-targets-cnt + 1 )
            ]
          ]
        ]
        this-item = "@" [
          ask patch x y [
            sprout-workers 1
          ]
        ]
        this-item = "$" [
          if placed-boxes-cnt < max-targets [
            ask patch x y [
              sprout-boxes 1
              set is-empty? false
              set placed-boxes-cnt ( placed-boxes-cnt + 1 )
          ]
          ]
      ])
    ]
  ]
  get-shapes
  ask patches with [ pcolor != grey ] [ ;; give ids to patches
    set id first id-list
    set id-list bf id-list
  ]
  ask targets [ set on-position [ id ] of patch-here ]
end


to go
  if steps = max-steps-per-episode [ reset-episode ]

  if episode-cnt = 100 [
    set episode-cnt 0
    set epoch-cnt ( epoch-cnt + 1 )
    set reward-last-epoch ( accumulated-reward / 100 )
    set accumulated-reward 0
  ]

  update-ids

  ask workers [
    ( qlearningextension:learning false)
  ]

  if max-boxes-on-target < f-cnt [ set max-boxes-on-target f-cnt ]
  if max-boxes-on-target-overall < f-cnt [ set max-boxes-on-target-overall f-cnt ]

  set steps ( steps + 1 )
  tick
end

to-report calculate-state
  let state-list []
  set state-list lput on-position state-list
  foreach boxes-order [
    box2 ->
    ask box2 [ set state-list lput on-position state-list ]
  ]
  ;; output-print state-list
  report state-list
end

;; update the positions of the boxes and the worker
to update-ids
  ask boxes [
    set last-position on-position
    set on-position [ id ] of patch-here
  ]
  ask workers [
    set on-position [ id ] of patch-here
  ]

end



to move-up
  ask workers [
    set heading 0
    ( ifelse
      ( not any? walls-on patch-ahead 1 ) and ( not any? boxes-on patch-ahead 1 ) [ fd 1  ]
      ( not any? walls-on patch-ahead 1 ) and ( not any? walls-on patch-ahead 2 ) and ( any? boxes-on patch-ahead 1 ) and ( not any? boxes-on patch-ahead 2 ) [
        ask boxes-on patch-ahead 1 [ set heading 0 fd 1 ]
        fd 1
      ]
    )
    update-ids
    set current-state calculate-state
  ]
end

to move-left
  ask workers [
    set heading 270
    ( ifelse
      ( not any? walls-on patch-ahead 1 ) and ( not any? boxes-on patch-ahead 1 ) [ fd 1  ]
      ( not any? walls-on patch-ahead 1 ) and ( not any? walls-on patch-ahead 2 ) and ( any? boxes-on patch-ahead 1 ) and ( not any? boxes-on patch-ahead 2 ) [
        ask boxes-on patch-ahead 1 [ set heading 270 fd 1 ]
        fd 1
      ]
    )
    update-ids
    set current-state calculate-state
  ]
end

to move-right
  ask workers [
    set heading 90
   ( ifelse
      ( not any? walls-on patch-ahead 1 ) and ( not any? boxes-on patch-ahead 1 ) [ fd 1 ]
      ( not any? walls-on patch-ahead 1 ) and ( not any? walls-on patch-ahead 2 ) and ( any? boxes-on patch-ahead 1 ) and ( not any? boxes-on patch-ahead 2 ) [
        ask boxes-on patch-ahead 1 [ set heading 90 fd 1 ]
        fd 1
      ]
    )
    update-ids
    set current-state calculate-state
  ]
end

to move-down
  ask workers [
    set heading 180
    ( ifelse
      ( not any? walls-on patch-ahead 1 ) and ( not any? boxes-on patch-ahead 1 ) [ fd 1 ]
      ( not any? walls-on patch-ahead 1 ) and ( not any? walls-on patch-ahead 2 ) and ( any? boxes-on patch-ahead 1 ) and ( not any? boxes-on patch-ahead 2 ) [
        ask boxes-on patch-ahead 1 [ set heading 180 fd 1  ]
        fd 1
      ]
    )
    update-ids
    set current-state calculate-state
  ]
end

to-report is-finished?
  let finished-count 0
  ask targets [ if any? boxes-here [ set finished-count ( finished-count + 1 ) ] ]
  set f-cnt finished-count
  ifelse finished-count = target-count [
    set times-finished (times-finished + 1)
    set times-finished-epoch ( times-finished-epoch + 1)
    report true ] [ report false ]
end

to check-deadlock
  set deadlock is-deadlock?
end

to-report is-deadlock?
  (ifelse
    boxes-stuck-in-2x2 = true [ set deadlock true report true]
    [ set deadlock false report false ]
  )
end

to-report boxes-stuck-in-2x2
  let occupied2x2 false
  ask boxes [
    if ( not any? targets-here and (( (( any? boxes-on patch-at -1 0 ) or ( any? walls-on patch-at -1 0 )) and (( any? boxes-on patch-at -1 1 ) or ( any? walls-on patch-at -1 1 )) and (( any? boxes-on patch-at 0 1 ) or ( any? walls-on patch-at 0 1 ))) ) or
      ( not any? targets-here and ((( any? boxes-on patch-at -1 0 ) or ( any? walls-on patch-at -1 0 )) and (( any? boxes-on patch-at -1 -1 ) or ( any? walls-on patch-at -1 -1 )) and (( any? boxes-on patch-at 0 -1 ) or ( any? walls-on patch-at 0 -1 ))) ) or
      ( not any? targets-here and ((( any? boxes-on patch-at 1 0 ) or ( any? walls-on patch-at 1 0 )) and (( any? boxes-on patch-at 1 1 ) or ( any? walls-on patch-at 1 1 )) and (( any? boxes-on patch-at 0 1 ) or ( any? walls-on patch-at 0 1 ))) ) or
      ( not any? targets-here and ((( any? boxes-on patch-at 1 0 ) or ( any? walls-on patch-at 1 0 )) and (( any? boxes-on patch-at 1 -1 ) or ( any? walls-on patch-at 1 -1 )) and (( any? boxes-on patch-at 0 -1 ) or ( any? walls-on patch-at 0 -1 ))) ) ) [ set occupied2x2 true ]
  ]
  report occupied2x2
end

to-report reward
  set box-in-corner? is-deadlock?
  let current-reward 0
  (ifelse
    reward-model = 1 [
      (ifelse
        is-finished? = true [ set current-reward 1 ]
        box-in-corner? = true [ set current-reward -500 ]
        f-cnt-old > f-cnt [ set current-reward -0.5 ]
        f-cnt-old < f-cnt [ set current-reward 0.5 ]
        [ set current-reward -0.1 ]
      )
    ]
    reward-model = 2 [
      (ifelse
        is-finished? = true [ set current-reward 100 ]
        box-in-corner? = true [ set current-reward -500 ]
        f-cnt-old > f-cnt [ set current-reward -0.5 ]
        f-cnt-old < f-cnt [ set current-reward 0.5 ]
        [ set current-reward -0.1 ]
      )
    ]
    reward-model = 3 [
      set current-reward -0.1
      if is-finished? = true [ set current-reward 0 ]
      if box-in-corner? = true [ set current-reward -500 ]
    ]
    reward-model = 4 [
      set current-reward -0.1
      if is-finished? = true [ set current-reward 100 ]
      if box-in-corner? = true [ set current-reward -500 ]
    ]
    reward-model = 5 [
      set current-reward -1
      if is-finished? = true [ set current-reward 10 ]
      if box-in-corner? = true [ set current-reward -5000 ]
    ]
    [ ;; reward-model 6 introduced for training with three targets to encourage agent not to stay in the same state
      set current-reward -1
      if is-finished? = true [ set current-reward 100 ]
      if box-in-corner? = true [ set current-reward -500 ]
      if same-state? [ set current-reward -10 ]
    ]
    )
  if is-deadlock? [ set total-deadlock-cnt ( total-deadlock-cnt + 1 ) ]
  set is-end-state? ( is-finished? or is-deadlock? or steps = max-steps-per-episode)
  set reward-current-episode reward-current-episode + current-reward
  set accumulated-reward accumulated-reward + current-reward
  set f-cnt-old f-cnt
  report current-reward
end

to-report same-state? ;; if no boxes are moved report true
  let same-state-cnt 0
  ask boxes [
    if on-position = last-position [ set same-state-cnt ( same-state-cnt + 1 ) ]
  ]
  ifelse same-state-cnt = max-targets [ report true ] [ report false ]
end

to reset-episode
  set episode-cnt ( episode-cnt + 1 )
  set max-boxes-on-target 0
  set is-end-state? false
  set reward-last-episode reward-current-episode
  set reward-current-episode 0
  set steps 0
  set box-in-corner? false
  set deadlock false
  ask boxes [ set xcor x-start set ycor y-start ]
  ask workers [ set xcor x-start set ycor y-start ]
end


to print-stats
  output-print csv:to-row ( list "learning rate" learning-rate )
  output-print csv:to-row ( list "step" steps )
  output-print csv:to-row ( list "episode" episode-cnt )
  output-print csv:to-row ( list "epoch" epoch-cnt )
end

;; used for setup
to get-shapes
  ask walls [
    set shape "tile brick"
    set color grey
  ]
  ask targets [
   set shape "target"
   set color red
  ]
  ask workers [
    set shape "person"
    set color green
  ]
  ask boxes [
    set color 7
    set shape "box"
  ]
end
@#$#@#$#@
GRAPHICS-WINDOW
351
10
659
319
-1
-1
30.0
1
10
1
1
1
0
1
1
1
0
9
0
9
0
0
1
ticks
30.0

BUTTON
87
91
150
124
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
67
190
147
223
NIL
move-up
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
18
230
103
263
NIL
move-left
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
112
230
204
263
NIL
move-right
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
62
270
158
303
NIL
move-down
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

MONITOR
81
366
219
411
Boxes on target
f-cnt
17
1
11

BUTTON
88
136
151
169
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

MONITOR
226
366
405
411
Max. boxes on target this episode
max-boxes-on-target
17
1
11

MONITOR
409
366
582
411
Max. boxes on target overall
max-boxes-on-target-overall
17
1
11

MONITOR
84
422
228
467
NIL
reward-current-episode
17
1
11

BUTTON
179
150
242
183
NIL
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

MONITOR
268
471
357
516
NIL
times-finished
17
1
11

MONITOR
587
365
665
410
NIL
episode-cnt
17
1
11

MONITOR
232
421
358
466
NIL
accumulated-reward
17
1
11

SLIDER
5
324
123
357
disc-factor
disc-factor
0.5
1
0.55
0.05
1
NIL
HORIZONTAL

SLIDER
126
324
241
357
learning-rate
learning-rate
0.05
1
0.7
0.05
1
NIL
HORIZONTAL

SLIDER
244
324
355
357
reward-model
reward-model
0
6
5.0
1
1
NIL
HORIZONTAL

MONITOR
7
421
76
466
NIL
epoch-cnt
17
1
11

BUTTON
187
273
303
306
NIL
check-deadlock
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

MONITOR
8
367
70
412
NIL
deadlock
17
1
11

MONITOR
6
471
150
516
NIL
reward-last-episode
17
1
11

MONITOR
157
470
263
515
NIL
reward-last-epoch
17
1
11

PLOT
371
417
672
537
Average Reward last 100 Episodes
Epoch
reward
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"pen-1" 1.0 0 -7500403 true "" "plot reward-last-epoch"

SLIDER
359
324
510
357
max-targets
max-targets
0
4
2.0
1
1
NIL
HORIZONTAL

SLIDER
514
324
699
357
max-steps-per-episode
max-steps-per-episode
150
1000
350.0
10
1
NIL
HORIZONTAL

@#$#@#$#@
## WHAT IS IT?

This model uses tabular Q-Learning to solve a Sokoban level

## HOW IT WORKS

The agent can move in a total of four directions and has to learn to push all the boxes onto the target locations.
The agent can only move onto empty floor tiles or push boxes onto empty floor tiles.
The logic for movement is implemented in the respective move functions.

While episodes refer to one run until the environment is reset, one epoch is 100 episodes. These metrics are provided using the Monitors.

## HOW TO USE IT

One can adjust the learning rate and the discount factor using the respective sliders.
Furthermore you can edit the reward model using the slider.
Additionally, the user can adjust the number of boxes loaded from the level_0.txt file
You can also replace this with another level from (https://github.com/google-deepmind/boxoban-levels), but you would need to bring it in the same format, e.g. one level per file, first line should be the level number and there should be an empty space before all the remaining lines.
However, using more than two targets will increase training time, computational resources and will probably lead to the agent never successfully solve the level.
Furthermore you can adjust the total number of steps using the slider provided.

You can also play by yourself by using the buttons provided and check if the current state is a deadlock by using the "check-deadlock" button.


## THINGS TO NOTICE

Notice the difference adjusting the learning rate, discount factor as well as the used reward model make.
Additional metrics like max-boxes-on-targets are implement for a training using two or more targets.
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tile brick
false
0
Rectangle -1 true false 0 0 300 300
Rectangle -7500403 true true 15 225 150 285
Rectangle -7500403 true true 165 225 300 285
Rectangle -7500403 true true 75 150 210 210
Rectangle -7500403 true true 0 150 60 210
Rectangle -7500403 true true 225 150 300 210
Rectangle -7500403 true true 165 75 300 135
Rectangle -7500403 true true 15 75 150 135
Rectangle -7500403 true true 0 0 60 60
Rectangle -7500403 true true 225 0 300 60
Rectangle -7500403 true true 75 0 210 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.2.2
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
