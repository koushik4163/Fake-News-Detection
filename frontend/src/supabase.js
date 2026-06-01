import { createClient } from '@supabase/supabase-js'

const SUPABASE_URL = "https://oskelanbekbwjenypytw.supabase.co"
const SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9za2VsYW5iZWtid2plbnlweXR3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODAyMTM3NTksImV4cCI6MjA5NTc4OTc1OX0.igtyw9MXmsTjtJTX4lX70ryCEd6BCUkuRFWTsIQxZfI"

export const supabase = createClient(SUPABASE_URL, SUPABASE_KEY)