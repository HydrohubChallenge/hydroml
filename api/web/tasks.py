from celery import shared_task


@shared_task
def add(x, y):
    res = x + y
    print(f'Adding {x} + {y} = {res}')
