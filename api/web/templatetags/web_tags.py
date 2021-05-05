from django import template
import datetime


register = template.Library()


@register.simple_tag
def duration(start_time, finish_time):
    return (finish_time - start_time) < datetime.timedelta(minutes=1)

