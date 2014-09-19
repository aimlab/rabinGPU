#ifndef TIMER_H
#define TIMER_H

typedef struct mytimer_s {
    double start_ms;
    double stop_ms;
} mytimer_t;

void timer_start(mytimer_t *timer);
void timer_stop(mytimer_t *timer);
double timer_elapsed(mytimer_t *timer);

#endif /* _TIMER_H */
